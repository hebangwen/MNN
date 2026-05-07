//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mls_server.hpp"
#include <iostream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <chrono>

namespace mls {

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static std::string GetCurrentTimeAsString() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    return std::to_string(seconds);
}

static std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });
    return std::string(it, str.end());
}

// ---------------------------------------------------------------------------
// Message conversion helpers
// ---------------------------------------------------------------------------

bool MessageToPromptItem(const json& j, PromptItem& item, int& token_count) {
    if (!j.is_object() || !j.contains("role")) {
        return false;
    }

    std::string role = j["role"].get<std::string>();

    // Tool message: must have tool_call_id and content
    if (role == "tool") {
        if (!j.contains("content") || !j.contains("tool_call_id")) {
            return false;
        }
        std::string content = j["content"].is_string()
            ? j["content"].get<std::string>()
            : j["content"].dump();
        // Use "json" role to pass the full tool message object to the template
        json tool_msg;
        tool_msg["role"] = "tool";
        tool_msg["content"] = content;
        tool_msg["tool_call_id"] = j["tool_call_id"];
        if (j.contains("name")) {
            tool_msg["name"] = j["name"];
        }
        item.first = "json";
        item.second = tool_msg.dump();
        token_count += static_cast<int>(content.size() / 4);  // rough estimate
        return true;
    }

    // Content may be string, null, or array of content parts
    std::string content;
    bool is_complex = false;
    json full_msg = j;

    if (j.contains("content")) {
        if (j["content"].is_null()) {
            content = "";
        } else if (j["content"].is_string()) {
            content = j["content"].get<std::string>();
        } else if (j["content"].is_array()) {
            // Array of content parts (e.g., text + image_url)
            content = j["content"].dump();
            is_complex = true;
        } else {
            content = j["content"].dump();
        }
    }

    // Assistant message with tool_calls
    if (role == "assistant" && j.contains("tool_calls")) {
        is_complex = true;
    }

    // Assistant message with reasoning_content
    if (j.contains("reasoning_content")) {
        is_complex = true;
    }

    if (is_complex) {
        // Use "json" role to pass full JSON to template
        item.first = "json";
        item.second = full_msg.dump();
    } else {
        item.first = role;
        item.second = content;
    }

    token_count += static_cast<int>(content.size() / 4);  // rough estimate
    return true;
}

// ---------------------------------------------------------------------------
// Tool call parsing from model output text
// ---------------------------------------------------------------------------

// Try to extract a JSON array or object from text between markers.
// Supports formats from common chat templates:
//   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
//   ```json\n{"name": "...", "arguments": {...}}\n```
//   {"name": "...", "arguments": {...}}
static std::string ExtractJsonFromText(const std::string& text) {
    // Pattern 1: <tool_call>...</tool_call>
    std::regex tool_call_re(R"(<tool_call>\s*(\{.*?\})\s*</tool_call>)", std::regex::icase);
    std::smatch match;
    if (std::regex_search(text, match, tool_call_re)) {
        return match[1].str();
    }

    // Pattern 2: ```json ... ```
    std::regex json_re(R"(```json\s*(\{.*?\})\s*```)", std::regex::icase);
    if (std::regex_search(text, match, json_re)) {
        return match[1].str();
    }

    // Pattern 3: ``` ... ``` (generic code block with JSON)
    std::regex code_re(R"(```\s*(\{.*?\})\s*```)");
    if (std::regex_search(text, match, code_re)) {
        auto inner = match[1].str();
        if (json::accept(inner)) {
            return inner;
        }
    }

    // Pattern 4: Find the last valid JSON object in the text
    // Try to find balanced braces
    auto first_brace = text.find('{');
    if (first_brace != std::string::npos) {
        auto last_brace = text.rfind('}');
        if (last_brace != std::string::npos && last_brace > first_brace) {
            auto candidate = text.substr(first_brace, last_brace - first_brace + 1);
            if (json::accept(candidate)) {
                return candidate;
            }
        }
    }

    return "";
}

json ParseToolCallsFromText(const std::string& text) {
    auto json_str = ExtractJsonFromText(text);
    if (json_str.empty()) {
        return json();  // null
    }

    auto parsed = json::parse(json_str, nullptr, false);
    if (parsed.is_discarded()) {
        return json();  // null
    }

    json tool_calls = json::array();

    // Single tool call object: {"name": "...", "arguments": {...}}
    if (parsed.is_object()) {
        json tc;
        tc["id"] = "call_" + GetCurrentTimeAsString();
        tc["type"] = "function";
        if (parsed.contains("name")) {
            tc["function"]["name"] = parsed["name"];
        }
        if (parsed.contains("arguments")) {
            if (parsed["arguments"].is_string()) {
                tc["function"]["arguments"] = parsed["arguments"].get<std::string>();
            } else {
                tc["function"]["arguments"] = parsed["arguments"].dump();
            }
        }
        tool_calls.push_back(tc);
    }
    // Array of tool calls: [{"name": "...", "arguments": {...}}, ...]
    else if (parsed.is_array()) {
        for (size_t i = 0; i < parsed.size(); i++) {
            if (!parsed[i].is_object()) continue;
            json tc;
            tc["id"] = "call_" + GetCurrentTimeAsString() + "_" + std::to_string(i);
            tc["type"] = "function";
            if (parsed[i].contains("name")) {
                tc["function"]["name"] = parsed[i]["name"];
            }
            if (parsed[i].contains("arguments")) {
                if (parsed[i]["arguments"].is_string()) {
                    tc["function"]["arguments"] = parsed[i]["arguments"].get<std::string>();
                } else {
                    tc["function"]["arguments"] = parsed[i]["arguments"].dump();
                }
            }
            tool_calls.push_back(tc);
        }
    }

    return tool_calls.is_null() || tool_calls.empty() ? json() : tool_calls;
}

json BuildResponseMessage(const std::string& answer,
                          std::string& content_out,
                          json& tool_calls_out) {
    auto tool_calls = ParseToolCallsFromText(answer);
    if (!tool_calls.is_null()) {
        content_out = "";
        tool_calls_out = tool_calls;
        json msg;
        msg["role"] = "assistant";
        msg["content"] = json(nullptr);
        msg["tool_calls"] = tool_calls;
        return msg;
    }

    content_out = answer;
    tool_calls_out = json();
    json msg;
    msg["role"] = "assistant";
    msg["content"] = content_out;
    return msg;
}

// ---------------------------------------------------------------------------
// Model info helper
// ---------------------------------------------------------------------------

std::string MlsServer::GetModelName(MNN::Transformer::Llm* llm) {
    auto config = llm->dump_config();
    auto cfg = json::parse(config, nullptr, false);
    if (!cfg.is_discarded()) {
        if (cfg.contains("model_name")) {
            return cfg["model_name"].get<std::string>();
        }
        if (cfg.contains("model_type")) {
            return cfg["model_type"].get<std::string>();
        }
    }
    return "mnn-model";
}

int MlsServer::EstimateTokenCount(const std::string& text) {
    // Rough estimate: ~4 chars per token for English/Chinese mixed text
    if (text.empty()) return 0;
    return std::max(1, static_cast<int>(text.size() / 4));
}

// ---------------------------------------------------------------------------
// Apply sampling / generation params from request to LLM runtime config
// ---------------------------------------------------------------------------

void MlsServer::ApplyRequestParams(MNN::Transformer::Llm* llm, const json& request_json) {
    json runtime_cfg;

    if (request_json.contains("temperature")) {
        runtime_cfg["temperature"] = request_json["temperature"].get<float>();
    }
    if (request_json.contains("top_p")) {
        runtime_cfg["top_p"] = request_json["top_p"].get<float>();
    }
    if (request_json.contains("top_k")) {
        runtime_cfg["topK"] = request_json["top_k"].get<int>();
    }
    if (request_json.contains("frequency_penalty")) {
        runtime_cfg["frequency_penalty"] = request_json["frequency_penalty"].get<float>();
    }
    if (request_json.contains("presence_penalty")) {
        runtime_cfg["presence_penalty"] = request_json["presence_penalty"].get<float>();
    }
    if (request_json.contains("repetition_penalty")) {
        runtime_cfg["repetition_penalty"] = request_json["repetition_penalty"].get<float>();
    }
    if (request_json.contains("max_tokens")) {
        runtime_cfg["max_new_tokens"] = request_json["max_tokens"].get<int>();
    }
    if (request_json.contains("seed")) {
        runtime_cfg["seed"] = request_json["seed"].get<int>();
    }

    if (!runtime_cfg.empty()) {
        llm->set_config(runtime_cfg.dump());
    }
}

// ---------------------------------------------------------------------------
// R1 model helpers
// ---------------------------------------------------------------------------

static const std::string getR1AssistantString(std::string assistant_content) {
    std::size_t pos = assistant_content.find("</think>");
    if (pos != std::string::npos) {
        assistant_content.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

static std::string GetR1UserString(std::string user_content, bool last) {
    return "<|User|>" + std::string(user_content) + "<|Assistant|>";
}

static std::vector<PromptItem> ConvertToR1(std::vector<PromptItem> chat_prompts) {
    std::vector<PromptItem> result_prompts = {};
    result_prompts.emplace_back("system", "<|begin_of_sentence|>You are a helpful assistant.");
    auto iter = chat_prompts.begin();
    for (; iter != chat_prompts.end() - 1; ++iter) {
        if (iter->first == "system") {
            continue;
        } else if (iter->first == "assistant") {
            result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
        } else if (iter->first == "user") {
            result_prompts.emplace_back("user", GetR1UserString(iter->second, false));
        }
    }
    if (iter->first == "user") {
        result_prompts.emplace_back("user", GetR1UserString(iter->second, true));
    } else {
        result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
    }
    return result_prompts;
}

// ---------------------------------------------------------------------------
// Non-streaming answer
// ---------------------------------------------------------------------------

void MlsServer::Answer(MNN::Transformer::Llm* llm, const json& messages,
                       const std::string& tools_json,
                       std::function<void(const std::string&, int prompt_tokens, int completion_tokens)> on_result) {
    ChatMessages prompts;
    int prompt_tokens = 0;
    if (messages.is_array()) {
        for (const auto& item_json : messages) {
            PromptItem item;
            int token_count = 0;
            if (!MessageToPromptItem(item_json, item, token_count)) {
                std::cerr << "Error converting JSON object to PromptItem." << std::endl;
                break;
            }
            prompts.push_back(item);
            prompt_tokens += token_count;
        }
    }

    std::stringstream response_buffer;
    int completion_tokens = 0;
    bool done = false;
    Utf8StreamProcessor processor([&response_buffer, on_result, &completion_tokens, prompt_tokens, &done](const std::string& utf8Char) {
        bool is_eop = utf8Char.find("<eop>") != std::string::npos;
        if (!is_eop) {
            response_buffer << utf8Char;
            completion_tokens++;
        } else {
            std::string response_result = response_buffer.str();
            on_result(response_result, prompt_tokens, completion_tokens);
            done = true;
        }
    });
    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len) {
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);
    {
        std::lock_guard<std::mutex> lock(llm_mutex_);
        if (is_r1_) {
            llm->response(ConvertToR1(prompts), &output_ostream, "<eop>");
        } else if (!tools_json.empty()) {
            llm->response(prompts, tools_json, &output_ostream, "<eop>");
        } else {
            llm->response(prompts, &output_ostream, "<eop>");
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming answer
// ---------------------------------------------------------------------------

void MlsServer::AnswerStreaming(MNN::Transformer::Llm* llm,
                                const json& messages,
                                const std::string& tools_json,
                                std::function<void(const std::string&, bool end, int prompt_tokens, int completion_tokens)> on_partial) {
    ChatMessages prompts;
    int prompt_tokens = 0;
    if (messages.is_array()) {
        for (const auto& item_json : messages) {
            PromptItem item;
            int token_count = 0;
            if (!MessageToPromptItem(item_json, item, token_count)) {
                std::cerr << "Error converting JSON object to PromptItem." << std::endl;
                return;
            }
            prompts.push_back(item);
            prompt_tokens += token_count;
        }
    }

    std::string answer = "";
    auto completion_tokens = std::make_shared<int>(0);
    Utf8StreamProcessor processor([&on_partial, &answer, completion_tokens](const std::string &utf8Char) {
        bool is_eop = (utf8Char.find("<eop>") != std::string::npos);
        if (is_eop) {
            on_partial("", true, 0, *completion_tokens);
        } else {
            answer += utf8Char;
            (*completion_tokens)++;
            on_partial(utf8Char, false, 0, 0);
        }
    });

    LlmStreamBuffer stream_buffer([&processor](const char* str, size_t len) {
        processor.processStream(str, len);
    });
    std::ostream output_ostream(&stream_buffer);
    {
        std::lock_guard<std::mutex> lock(llm_mutex_);
        if (is_r1_) {
            llm->response(ConvertToR1(prompts), &output_ostream, "<eop>");
        } else if (!tools_json.empty()) {
            llm->response(prompts, tools_json, &output_ostream, "<eop>");
        } else {
            llm->response(prompts, &output_ostream, "<eop>");
        }
    }
}

// ---------------------------------------------------------------------------
// CORS helper
// ---------------------------------------------------------------------------

static void AllowCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods",  "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers",  "Content-Type, Authorization");
}

// ---------------------------------------------------------------------------
// Chat completion request handler (shared by /chat/completions and /v1/...)
// ---------------------------------------------------------------------------

using ChatCompletionHandler = std::function<void(const httplib::Request&, httplib::Response&, MNN::Transformer::Llm*, bool)>;

static ChatCompletionHandler MakeChatCompletionHandler(MlsServer* self) {
    return [self](const httplib::Request& req, httplib::Response& res,
                  MNN::Transformer::Llm* llm, bool is_r1) {
        AllowCors(res);

        if (!json::accept(req.body)) {
            json err;
            err["error"]["message"] = "Invalid JSON in request body.";
            err["error"]["type"] = "invalid_request_error";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        json request_json = json::parse(req.body, nullptr, false);
        json messages = request_json["messages"];
        std::string model = request_json.value("model", self->GetModelName(llm));
        bool stream = request_json.value("stream", false);

        // Extract tools from request
        std::string tools_json;
        if (request_json.contains("tools")) {
            tools_json = request_json["tools"].dump();
        }

        // Apply sampling / generation params from request
        self->ApplyRequestParams(llm, request_json);

        // Estimate prompt tokens from messages
        int estimated_prompt_tokens = self->EstimateTokenCount(messages.dump());

        if (!stream) {
            // --- Non-streaming response ---
            std::string full_answer;
            int prompt_tokens = estimated_prompt_tokens;
            int completion_tokens = 0;

            self->Answer(llm, messages, tools_json,
                [&](const std::string& answer, int pt, int ct) {
                    full_answer = answer;
                    if (pt > 0) prompt_tokens = pt;
                    completion_tokens = ct;
                });

            // Build response message with tool_calls extraction
            std::string content;
            json tool_calls;
            auto response_msg = BuildResponseMessage(full_answer, content, tool_calls);

            json response_json = {
                {"id", "chatcmpl-" + GetCurrentTimeAsString()},
                {"object", "chat.completion"},
                {"created", static_cast<int>(time(nullptr))},
                {"model", model},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", response_msg},
                        {"finish_reason", "stop"}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens}
                }},
                {"system_fingerprint", "mnn-agent"}
            };
            res.set_content(response_json.dump(), "application/json");
            return;
        }

        // --- Streaming (SSE) response ---
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        int pt = estimated_prompt_tokens;
        int ct = 0;

        res.set_chunked_content_provider(
            "text/event-stream",
            [self, llm, messages, tools_json, model, pt, &ct](size_t, httplib::DataSink &sink) mutable {
                auto sse_callback = [&](const std::string &partial_text, bool end,
                                        int /*prompt_tokens*/, int completion_tokens) {
                    if (completion_tokens > 0) ct = completion_tokens;
                    std::string finish_reason = end ? "stop" : "";

                    json sse_json = {
                        {"id", "chatcmpl-" + GetCurrentTimeAsString()},
                        {"object", "chat.completion.chunk"},
                        {"created", static_cast<int>(std::time(nullptr))},
                        {"model", model},
                        {"choices", json::array({
                            {
                                {"delta", {{"content", partial_text}}},
                                {"index", 0},
                                {"finish_reason", finish_reason}
                            }
                        })}
                    };

                    if (end) {
                        sse_json["usage"] = {
                            {"prompt_tokens", pt},
                            {"completion_tokens", ct},
                            {"total_tokens", pt + ct}
                        };
                    }

                    std::string chunk_str = "data: " + sse_json.dump() + "\n\n";
                    sink.os.write(chunk_str.c_str(), chunk_str.size());
                    sink.os.flush();
                };
                self->AnswerStreaming(llm, messages, tools_json, sse_callback);
                std::string done_str = "data: [DONE]\n\n";
                sink.os.write(done_str.c_str(), done_str.size());
                sink.os.flush();
                sink.done();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return false;
            });
    };
}

// ---------------------------------------------------------------------------
// Server start
// ---------------------------------------------------------------------------

void MlsServer::Start(MNN::Transformer::Llm* llm, bool is_r1) {
    this->is_r1_ = is_r1;
    httplib::Server server;

    // GET / — HTML chat UI
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        res.set_content(html_content, "text/html");
    });

    // GET /health — health check
    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        AllowCors(res);
        json health;
        health["status"] = "ok";
        res.set_content(health.dump(), "application/json");
    });

    // GET /v1/models — OpenAI-compatible model listing
    server.Get("/v1/models", [this, llm](const httplib::Request&, httplib::Response& res) {
        AllowCors(res);
        std::string model_name = GetModelName(llm);
        json models;
        models["object"] = "list";
        models["data"] = json::array({
            {
                {"id", model_name},
                {"object", "model"},
                {"created", static_cast<int>(time(nullptr))},
                {"owned_by", "mnn"}
            }
        });
        res.set_content(models.dump(), "application/json");
    });

    // POST /reset — reset LLM state
    server.Post("/reset", [&](const httplib::Request &req, httplib::Response &res) {
        AllowCors(res);
        llm->reset();
        res.set_content("{\"status\": \"ok\"}", "application/json");
    });

    // OPTIONS — CORS preflight for both chat endpoints
    server.Options("/chat/completions", [](const httplib::Request&, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    server.Options("/v1/chat/completions", [](const httplib::Request&, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });

    auto handler = MakeChatCompletionHandler(this);

    // POST /chat/completions — legacy path (backward compatible)
    server.Post("/chat/completions", [handler, llm, is_r1](const httplib::Request &req, httplib::Response &res) {
        handler(req, res, llm, is_r1);
    });

    // POST /v1/chat/completions — standard OpenAI API path
    server.Post("/v1/chat/completions", [handler, llm, is_r1](const httplib::Request &req, httplib::Response &res) {
        handler(req, res, llm, is_r1);
    });

    std::cout << "Starting MNN MLS server on http://localhost:9090" << std::endl;
    std::cout << "  Chat:           POST /chat/completions" << std::endl;
    std::cout << "  Chat (v1):      POST /v1/chat/completions" << std::endl;
    std::cout << "  Models:         GET /v1/models" << std::endl;
    std::cout << "  Health:         GET /health" << std::endl;
    if (!server.listen("0.0.0.0", 9090)) {
        std::cerr << "Error: Could not start server." << std::endl;
    }
}

} // namespace mls
