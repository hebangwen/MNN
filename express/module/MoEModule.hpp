//
//  MoEModule.hpp
//  MNN
//
//  Created by MNN on 2025/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MoEModule_hpp
#define MoEModule_hpp

#include <functional>
#include <fstream>
#include <set>
#include <string>
#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"
namespace MNN {
namespace Express {

// Callback signature for MoE expert routing debug/verification.
// layerId: which transformer layer (0-indexed)
// numExperts: total number of experts in this MoE layer
// topK: how many experts each token activates
// seqLen: number of tokens being processed
// selectedExperts: [seqLen * topK] flattened, per-token topK expert IDs
// routingWeights: [seqLen * topK] flattened, per-token topK routing weights
using MoERoutingCallback = std::function<void(int layerId, int numExperts, int topK,
    int seqLen, const int* selectedExperts, const float* routingWeights)>;

class MoEModule : public Module {
public:
    virtual ~MoEModule() {} // Do nothing
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    static MoEModule* create(const Op* op, const std::map<std::string, SubGraph>& subGraph, std::shared_ptr<Executor::RuntimeManager> rtmgr, const Module::Config& config);
    
    // Set a global routing callback. All MoE layers will invoke this when routing.
    // Pass nullptr to clear. Thread-safe: only call during initialization, not concurrent forward.
    static void setRoutingCallback(MoERoutingCallback cb);
    
    // Set a directory for file-based dump of ALL tokens' routing data (experts + weights).
    // When non-empty, each MoE layer writes prefill data once, then appends decode data.
    // Pass empty string to disable. Clears sPrefillWritten on set.
    static void setDumpDir(const std::string& dir);
    
private:
    MoEModule(){}
    Module* clone(CloneContext* ctx) const override;
    int mNumExperts = 128, mTopK = 8, mLayerId = 0;
    std::vector<std::shared_ptr<Module>> mExperts;
    std::vector<VARP> mHiddenStatesList;
    
    static MoERoutingCallback sRoutingCallback;
    static std::string sDumpDir;           // directory for dump files
    static std::set<int> sPrefillWritten;  // layers that have written prefill data
};
}
}

#endif /* MoEModule_hpp */
