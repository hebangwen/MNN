//
//  MoEModule.cpp
//  MNN
//
//  Created by MNN on 2025/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MoEModule.hpp"
#include "PipelineModule.hpp"
#include "MNN_generated.h"
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace Express {

MoERoutingCallback MoEModule::sRoutingCallback = nullptr;
std::string MoEModule::sDumpDir;
std::set<int> MoEModule::sPrefillWritten;

void MoEModule::setRoutingCallback(MoERoutingCallback cb) {
    sRoutingCallback = cb;
}

void MoEModule::setDumpDir(const std::string& dir) {
    sDumpDir = dir;
    sPrefillWritten.clear();
}

std::vector<Express::VARP> MoEModule::onForward(const std::vector<Express::VARP>& inputs) {
    auto hiddenStates = inputs[0];
    auto routingWeights = inputs[1];
    auto selectedExperts = inputs[2];
    auto selectedDim = selectedExperts->getInfo()->dim;
    int ranks = static_cast<int>(selectedDim.size());
    const int seqlen = selectedDim[ranks - 2];
    const int topK   = selectedDim[ranks - 1];
    MNN_ASSERT(topK == mTopK);
    auto selectedPtr = selectedExperts->readMap<int>();
    auto routingPtr = routingWeights->readMap<float>();
    // Invoke routing callback (if set) for expert activation verification
    if (sRoutingCallback) {
        sRoutingCallback(mLayerId, mNumExperts, topK, seqlen, selectedPtr, routingPtr);
    }
    // File-based dump of ALL tokens for MoE routing verification
    if (!sDumpDir.empty()) {
        if (seqlen > 1) {
            bool isFirstPrefill = sPrefillWritten.empty();
            bool layerNeedsPrefill = sPrefillWritten.find(mLayerId) == sPrefillWritten.end();
            if (layerNeedsPrefill) {
                std::string expPath = sDumpDir + "/moe_layer_" + std::to_string(mLayerId) + "_prefill_experts.bin";
                std::string wgtPath = sDumpDir + "/moe_layer_" + std::to_string(mLayerId) + "_prefill_weights.bin";
                std::ofstream ef(expPath, std::ios::binary);
                std::ofstream wf(wgtPath, std::ios::binary);
                ef.write(reinterpret_cast<const char*>(selectedPtr), (std::streamsize)(seqlen * topK * sizeof(int)));
                wf.write(reinterpret_cast<const char*>(routingPtr), (std::streamsize)(seqlen * topK * sizeof(float)));
                sPrefillWritten.insert(mLayerId);
                if (isFirstPrefill) {
                    std::string metaPath = sDumpDir + "/moe_meta.json";
                    std::ofstream mf(metaPath);
                    mf << "{\"top_k\": " << topK
                       << ", \"num_experts\": " << mNumExperts
                       << ", \"layer_id\": " << mLayerId << "}";
                }
            }
        } else if (sPrefillWritten.find(mLayerId) != sPrefillWritten.end()) {
            std::string expPath = sDumpDir + "/moe_layer_" + std::to_string(mLayerId) + "_decode_experts.bin";
            std::string wgtPath = sDumpDir + "/moe_layer_" + std::to_string(mLayerId) + "_decode_weights.bin";
            std::ofstream ef(expPath, std::ios::binary | std::ios::app);
            std::ofstream wf(wgtPath, std::ios::binary | std::ios::app);
            ef.write(reinterpret_cast<const char*>(selectedPtr), (std::streamsize)(seqlen * topK * sizeof(int)));
            wf.write(reinterpret_cast<const char*>(routingPtr), (std::streamsize)(seqlen * topK * sizeof(float)));
        }
    }
#ifndef MNN_MOE_DEBUG_OFF
    // DEBUG: Print expert routing
    MNN_PRINT("[MoEDebug] layer=%d num_experts=%d topK=%d seqlen=%d\n", mLayerId, mNumExperts, topK, seqlen);
    for (int t = 0; t < seqlen && t < 4; t++) {
        MNN_PRINT("[MoEDebug]   token[%d]:", t);
        for (int k = 0; k < topK; k++) {
            int eid = selectedPtr[t * topK + k];
            float w = routingPtr[t * topK + k];
            MNN_PRINT(" expert=%d(%.4f)", eid, w);
        }
        MNN_PRINT("\n");
    }
    if (seqlen > 4) {
        MNN_PRINT("[MoEDebug]   ... (%d more tokens)\n", seqlen - 4);
    }
#endif
    // decode
#if 0 // using Expr for debug or clip some expert
    if (seqlen == 1) {
        auto routingPtr  = routingWeights->readMap<float>();
        int expertId = selectedPtr[0];
        float scale = routingPtr[0];
        auto output = mExperts[expertId]->onForward({hiddenStates})[0];
        auto finalHiddenStates = _Multiply(output, _Scalar<float>(scale));
        for (int i = 1; i < topK; ++i) {
            expertId = selectedPtr[i];
            scale = routingPtr[i];
            // if (scale < 0.1) {
            //     continue;
            // }
            output = mExperts[expertId]->onForward({hiddenStates})[0];
            auto curHiddenStates = _Multiply(output, _Scalar<float>(scale));
            finalHiddenStates = _Add(finalHiddenStates, curHiddenStates);
        }
        return {finalHiddenStates};
    }
#else
    if (seqlen == 1) {
        mHiddenStatesList.resize(topK+1);
        for (int i = 0; i < topK; ++i) {
            int expertId = selectedPtr[i];
            mHiddenStatesList[i] = mExperts[expertId]->onForward({hiddenStates})[0];
        }
        mHiddenStatesList[topK] = routingWeights;
        auto res = mExperts.back()->onForward(mHiddenStatesList);
        for (auto& p : mHiddenStatesList) {
            p = nullptr;
        }
        return res;
    }
#endif
    // prefill
    std::vector<std::vector<std::pair<int, float>>> expertWorks(mNumExperts, std::vector<std::pair<int, float>>());
    for (int i = 0; i < seqlen; ++i) {
        for (int j = 0; j < topK; ++j) {
            int expertId = selectedPtr[i * topK + j];
            int tokenId = i;
            float scale = routingPtr[i * topK + j];
            std::pair<int, float> tokenIdScale(tokenId, scale);
            expertWorks[expertId].push_back(tokenIdScale);
        }
    }
    auto sizeSplits = std::vector<int>(seqlen, 1);
    VARPS tokenHiddenStates = _Split(hiddenStates, sizeSplits, 0);
    VARPS finalHiddenStates(seqlen, VARP(nullptr));
    for (int i = 0; i < mNumExperts; ++i) {
        if (expertWorks[i].empty()) {
            continue;
        }
        if (expertWorks[i].size() > 1) {
            VARPS inputTokens;
            for (auto& tokenId : expertWorks[i]) {
                inputTokens.emplace_back(tokenHiddenStates[tokenId.first]);
            }
            VARP workHiddenStates = _Concat(inputTokens, 0);
            auto curHiddenStates = mExperts[i]->onForward({workHiddenStates})[0];
            VARPS curHiddenStatesList = _Split(curHiddenStates, std::vector<int>(expertWorks[i].size(), 1), 0);
            for (int j = 0; j < expertWorks[i].size(); ++j) {
                int tokenId = expertWorks[i][j].first;
                float scale = expertWorks[i][j].second;
                auto scaleHiddenStates = _Multiply(curHiddenStatesList[j], _Scalar<float>(scale));
                if (finalHiddenStates[tokenId] == nullptr) {
                    finalHiddenStates[tokenId] = scaleHiddenStates;
                } else {
                    finalHiddenStates[tokenId] = _Add(finalHiddenStates[tokenId], scaleHiddenStates);
                }
            }
        } else {
            int tokenId = expertWorks[i][0].first;
            float scale = expertWorks[i][0].second;
            VARP workHiddenStates = tokenHiddenStates[tokenId];
            auto output = mExperts[i]->onForward({workHiddenStates})[0];
            auto curHiddenStates = _Multiply(output, _Scalar<float>(scale));
            if (finalHiddenStates[tokenId] == nullptr) {
                finalHiddenStates[tokenId] = curHiddenStates;
            } else {
                finalHiddenStates[tokenId] = _Add(finalHiddenStates[tokenId], curHiddenStates);
            }
        }
    }
    auto output = _Concat(finalHiddenStates, 0);
    return {output};
}

MoEModule* MoEModule::create(const Op* op, const std::map<std::string, SubGraph>& subGraph, std::shared_ptr<Executor::RuntimeManager> rtmgr, const Module::Config& config) {
    auto module = new MoEModule;
    module->setType("MoEModule");
    auto moeParam = op->main_as_Extra();
    int numExperts = 128, topK = 1, layerId = 0;
    if (nullptr != moeParam->attr()) {
        for (int i = 0; i < moeParam->attr()->size(); ++i) {
            auto attr = moeParam->attr()->GetAs<Attribute>(i);
            if (nullptr != attr->key()) {
                if (attr->key()->str() == "num_experts") {
                    numExperts = attr->i();
                } else if (attr->key()->str() == "top_k") {
                    topK = attr->i();
                } else if (attr->key()->str() == "layer_id") {
                    layerId = attr->i();
                }
            }
        }
    }
    module->mNumExperts  = numExperts;
    module->mTopK        = topK;
    module->mLayerId     = layerId;
    for (int i = 0; i < numExperts; ++i) {
        std::string expertName = "/expert/" + std::to_string(layerId) + "_" + std::to_string(i);
        auto& expertG = subGraph.find(expertName)->second;
        module->mExperts.push_back(expertG.m);
    }
    if (nullptr != op->name()) {
        module->setName(op->name()->str());
    }
    // create a compute submodule
    {
        std::vector<std::string> inputNames;
        VARPS hidden_states_list;
        for (int i = 0; i < topK; ++i) {
            std::string name = std::to_string(i);
            auto inp = _Input({1, -1}, NCHW);
            inp->setName(name);
            hidden_states_list.emplace_back(inp);
            inputNames.emplace_back(name);
        }
        auto scales = _Input({1, 1, topK}, NCHW);
        scales->setName("scale");
        inputNames.emplace_back("scale");
        auto hidden_states = _Concat(hidden_states_list, 0);
        scales = _Reshape(scales, {-1, 1});
        hidden_states = _Multiply(hidden_states, scales);
        hidden_states = _ReduceSum(hidden_states, {0}, true);
        hidden_states->setName("o");
        auto netbuffer = Express::Variable::save({hidden_states});
        module->mExperts.emplace_back(PipelineModule::load(inputNames, {"o"}, (const uint8_t*)netbuffer.data(), netbuffer.size(), rtmgr, &config));
    }
    return module;
}

Module* MoEModule::clone(CloneContext* ctx) const {
    MoEModule* module(new MoEModule);
    for (auto& expert : mExperts) {
        module->mExperts.emplace_back(expert->clone(ctx));
    }
    module->mNumExperts = mNumExperts;
    module->mTopK = mTopK;
    module->mLayerId = mLayerId;
    return this->cloneBaseTo(ctx, module);
}

}  // namespace Express
}  // namespace MNN
