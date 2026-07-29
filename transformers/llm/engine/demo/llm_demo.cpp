//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>
#include <vector>
#include <string>
#if defined(LLM_SUPPORT_VISION) && defined(MNN_IMGCODECS)
#include <cv/cv.hpp>
#endif
//#define LLM_SUPPORT_AUDIO
#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif
using namespace MNN::Transformer;

// Callback to print activated experts per MoE layer
// Registered via Llm::setMoERoutingCallback()
static void expertRoutingCallback(int layerId, int numExperts, int topK,
                                   int seqLen, const int* selected,
                                   const float* weights) {
    // Count expert usage across all tokens
    std::vector<int> expertCounts(numExperts, 0);
    for (int t = 0; t < seqLen; t++) {
        for (int k = 0; k < topK; k++) {
            int eid = selected[t * topK + k];
            if (eid >= 0 && eid < numExperts) {
                expertCounts[eid]++;
            }
        }
    }
    // Print layer summary
    MNN_PRINT("[ExpertRouting] Layer %d: %d tokens, top-%d, %d experts\n",
              layerId, seqLen, topK, numExperts);
    // Find active experts and top-5 most used
    int activeExperts = 0;
    for (int i = 0; i < numExperts; i++) {
        if (expertCounts[i] > 0) activeExperts++;
    }
    MNN_PRINT("[ExpertRouting]   Active: %d/%d experts", activeExperts, numExperts);
    // Manual top-5 selection
    int topIds[5] = {-1, -1, -1, -1, -1};
    int topVals[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < numExperts; i++) {
        if (expertCounts[i] == 0) continue;
        for (int j = 0; j < 5; j++) {
            if (expertCounts[i] > topVals[j]) {
                for (int k = 4; k > j; k--) {
                    topIds[k] = topIds[k-1];
                    topVals[k] = topVals[k-1];
                }
                topIds[j] = i;
                topVals[j] = expertCounts[i];
                break;
            }
        }
    }
    for (int i = 0; i < 5 && topIds[i] >= 0; i++) {
        MNN_PRINT(" e%d:%d", topIds[i], topVals[i]);
    }
    MNN_PRINT("\n");
    // Print per-token routing for first 4 tokens
    for (int t = 0; t < seqLen && t < 4; t++) {
        MNN_PRINT("[ExpertRouting]   token[%d]:", t);
        for (int k = 0; k < topK; k++) {
            int eid = selected[t * topK + k];
            float w = weights[t * topK + k];
            MNN_PRINT(" e%d(%.3f)", eid, w);
        }
        MNN_PRINT("\n");
    }
}

static void tuning_prepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    int64_t sample_time = 0;
    // llm->warmup();
    auto context = llm->getContext();
    if (max_token_number > 0) {
        llm->set_config("{\"max_new_tokens\":1}");
    }
#ifdef LLM_SUPPORT_AUDIO
    std::vector<float> waveform;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
        waveform.reserve(waveform.size() + size);
        waveform.insert(waveform.end(), ptr, ptr + size);
        if (last_chunk) {
            auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
            MNN::AUDIO::save("output.wav", waveform_var, 24000);
            waveform.clear();
        }
        return true;
    });
#endif
    for (int i = 0; i < prompts.size(); i++) {
        auto prompt = prompts[i];
     // #define MIMO_NO_THINKING
     #ifdef MIMO_NO_THINKING
        // update config.json and llm_config.json if need. example:
        llm->set_config("{\"assistant_prompt_template\":\"<|im_start|>assistant\\n<think>\\n</think>\%s<|im_end|>\\n\"}");
        prompt = prompt + "<think>\n</think>";
     #endif

        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        
        if (max_token_number >= 0) {
            llm->response(prompt, &std::cout, nullptr, 0);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
                // Check for errors
                if(context->status == LlmStatus::INTERNAL_ERROR) {
                    MNN_ERROR("Error: Generation failed due to internal error\n");
                    return -1;
                }
            }
        } else {
            llm->response(prompt);
            // Check for errors after response
            if(context->status == LlmStatus::INTERNAL_ERROR) {
                MNN_ERROR("Error: Response generation failed due to internal error\n");
                return -1;
            }
        }
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
        sample_time += context->sample_us;
    }
    llm->generateWavform();

    float vision_s = context->vision_us / 1e6;
    float audio_s = context->audio_us / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    float sample_s = sample_time / 1e6;
    float vision_speed = 0.0f;
    if (context->pixels_mp > 0.0f) {
        vision_speed = context->pixels_mp / vision_s;
    }
    float audio_speed = 0.0f;
    if (context->audio_input_s > 0.0f) {
        audio_speed = context->audio_input_s / audio_s;
    }
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("prompt tokens num = %d\n", prompt_len);
    MNN_PRINT("decode tokens num = %d\n", decode_len);
    MNN_PRINT(" vision time = %.2f s\n", vision_s);
    MNN_PRINT(" pixels_mp = %.2f MP\n", context->pixels_mp);
    MNN_PRINT("  audio process time = %.2f s\n", audio_s);
    MNN_PRINT("  audio input time = %.2f s\n", context->audio_input_s);
    MNN_PRINT("prefill time = %.2f s\n", prefill_s);
    MNN_PRINT(" decode time = %.2f s\n", decode_s);
    MNN_PRINT(" sample time = %.2f s\n", sample_s);
    MNN_PRINT("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    MNN_PRINT(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    MNN_PRINT(" vision speed = %.3f MP/s\n", vision_speed);
    MNN_PRINT(" audio RTF = %.3f \n", audio_s / context->audio_input_s);
    MNN_PRINT("##################################\n");
    return 0;
}

static int ceval(Llm* llm, const std::vector<std::string>& lines, std::string filename) {
    auto csv_data = parse_csv(lines);
    int right = 0, wrong = 0;
    std::vector<std::string> answers;
    for (int i = 1; i < csv_data.size(); i++) {
        const auto& elements = csv_data[i];
        std::string prompt = elements[1];
        prompt += "\n\nA. " + elements[2];
        prompt += "\nB. " + elements[3];
        prompt += "\nC. " + elements[4];
        prompt += "\nD. " + elements[5];
        prompt += "\n\n";
        MNN_PRINT("%s", prompt.c_str());
        MNN_PRINT("## 进度: %d / %lu\n", i, lines.size() - 1);
        std::ostringstream lineOs;
        llm->response(prompt.c_str(), &lineOs);
        auto line = lineOs.str();
        MNN_PRINT("%s", line.c_str());
        answers.push_back(line);
    }
    {
        auto position = filename.rfind("/");
        if (position != std::string::npos) {
            filename = filename.substr(position + 1, -1);
        }
        position = filename.find("_val");
        if (position != std::string::npos) {
            filename.replace(position, 4, "_res");
        }
        std::cout << "store to " << filename << std::endl;
    }
    std::ofstream ofp(filename);
    ofp << "id,answer" << std::endl;
    for (int i = 0; i < answers.size(); i++) {
        auto& answer = answers[i];
        ofp << i << ",\""<< answer << "\"" << std::endl;
    }
    ofp.close();
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
//#define LLM_DEMO_ONELINE
#ifdef LLM_DEMO_ONELINE
    std::ostringstream tempOs;
    tempOs << prompt_fs.rdbuf();
    prompt = tempOs.str();
    prompts = {prompt};
#else
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.empty()) {
            continue;
        }
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
#endif
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    // ceval
    if (prompts[0] == "id,question,A,B,C,D,answer") {
        return ceval(llm, prompts, prompt_file);
    }
    return benchmark(llm, prompts, max_token_number);
}

void chat(Llm* llm) {
    ChatMessages messages;
    messages.emplace_back("system", "You are a helpful assistant.");
    auto context = llm->getContext();
    while (true) {
        std::cout << "\nUser: ";
        std::string user_str;
        std::getline(std::cin, user_str);
        if (user_str == "/exit") {
            return;
        }
        if (user_str == "/reset") {
            llm->reset();
            messages.clear();
            messages.emplace_back("system", "You are a helpful assistant.");
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        messages.emplace_back("user", user_str);
        std::cout << "\nA: " << std::flush;
        llm->response(messages);
        auto assistant_str = context->generate_str;
        messages.emplace_back("assistant", assistant_str);
    }
}

static void dumpOutputTensors(Llm* llm, const std::string& dump_dir) {
    auto outputs = llm->getOutputs();
    if (outputs.empty()) {
        MNN_PRINT("[dump] No output tensors to dump\n");
        return;
    }
    int logitsIdx = llm->getOutputIndex("logits");
    int hiddenIdx = llm->getOutputIndex("hidden_states");
    std::ofstream metaFile(dump_dir + "/mnn_meta.json");
    metaFile << "{\n";
    bool first = true;
    auto dumpOne = [&](const std::string& name, int idx) -> void {
        if (idx < 0 || idx >= (int)outputs.size()) return;
        auto& v = outputs[idx];
        auto info = v->getInfo();
        if (!info) return;
        auto ptr = v->readMap<float>();
        if (!ptr) return;
        std::ofstream f(dump_dir + "/" + name + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(ptr),
                (std::streamsize)(info->size * sizeof(float)));
        if (!first) metaFile << ",\n";
        metaFile << "  \"" << name << "_shape\": [";
        for (int i = 0; i < (int)info->dim.size(); i++) {
            if (i > 0) metaFile << ", ";
            metaFile << info->dim[i];
        }
        metaFile << "]";
        first = false;
    };
    dumpOne("logits", logitsIdx);
    dumpOne("hidden_states", hiddenIdx);
    metaFile << ",\n  \"note\": \"Outputs captured from last forward via getOutputs()\"\n}\n";
    MNN_PRINT("[dump] Output tensors written to %s\n", dump_dir.c_str());
}

static int runMultimodal(Llm* llm, const std::string& image_path,
                         const std::string& dump_dir, int max_token_number) {
#if defined(LLM_SUPPORT_VISION) && defined(MNN_IMGCODECS)
    MNN::Express::VARP image = MNN::CV::imread(image_path);
    if (image == nullptr) {
        MNN_ERROR("Failed to load image: %s\n", image_path.c_str());
        return -1;
    }
    MNN_PRINT("Loaded image: %s\n", image_path.c_str());
    MultimodalPrompt mp;
    mp.prompt_template = "<img>image</img>document parsing.";
    PromptImagePart imgPart;
    imgPart.image_data = image;
    imgPart.width = 0;
    imgPart.height = 0;
    mp.images["image"] = imgPart;
    llm->set_config(R"({"async":false})");
    auto context = llm->getContext();
    if (max_token_number > 0) {
        llm->response(mp, &std::cout, nullptr, 0);
        if (!dump_dir.empty()) {
            dumpOutputTensors(llm, dump_dir);
        }
        while (!llm->stoped() && context->gen_seq_len < max_token_number) {
            llm->generate(1);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                MNN_ERROR("Error: Generation failed due to internal error\n");
                return -1;
            }
        }
    } else {
        llm->response(mp);
        if (context->status == LlmStatus::INTERNAL_ERROR) {
            MNN_ERROR("Error: Response generation failed due to internal error\n");
            return -1;
        }
        if (!dump_dir.empty()) {
            dumpOutputTensors(llm, dump_dir);
        }
    }
    return 0;
#else
    (void)llm;
    (void)max_token_number;
    MNN_ERROR("Image loading not supported (need MNN_BUILD_OPENCV=ON and MNN_IMGCODECS=ON): %s\n",
              image_path.c_str());
    return -1;
#endif
}
int main(int argc, const char* argv[]) {
    std::string image_path, dump_dir;
    int max_token_number = -1;
    std::vector<std::string> positional_args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--dump_dir" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            max_token_number = std::stoi(argv[++i]);
        } else {
            positional_args.push_back(arg);
        }
    }

    if (positional_args.empty()) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt> [max_tokens]\n"
                  << "  --image <path>      Path to image file (multimodal input)\n"
                  << "  --dump_dir <path>   Directory for MoE + tensor dumps\n"
                  << "  --max_tokens <N>    Max tokens to generate\n";
        return 0;
    }

    std::string config_path = positional_args[0];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    if (!dump_dir.empty()) {
        Llm::setMoEDumpDir(dump_dir);
    }
    {
        AUTOTIME;
        bool res = llm->load();
        if (!res) {
            MNN_ERROR("LLM init error\n");
            return 0;
        }
    }
    // Register MoE expert routing callback for vision verification
    Llm::setMoERoutingCallback(expertRoutingCallback);
    if (dump_dir.empty()) {
        AUTOTIME;
        tuning_prepare(llm.get());
    }
    if (!image_path.empty()) {
        return runMultimodal(llm.get(), image_path, dump_dir, max_token_number);
    }
    if (positional_args.size() < 2) {
        chat(llm.get());
        return 0;
    }
    if (max_token_number < 0 && positional_args.size() >= 3) {
        std::istringstream os(positional_args[2]);
        os >> max_token_number;
    }
    if (positional_args.size() >= 4) {
        MNN_PRINT("Set not thinking, only valid for Qwen3\n");
        llm->set_config(R"({
            "jinja": {
                "context": {
                    "enable_thinking":false
                }
            }
        })");
    }
    std::string prompt_file = positional_args[1];
    llm->set_config(R"({
        "async":false
    })");
    return eval(llm.get(), prompt_file, max_token_number);
}
