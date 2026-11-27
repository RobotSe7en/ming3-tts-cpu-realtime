#pragma once

#include "kernels.h"
#include <memory>

namespace ming_cpu::trial {

class PackedWeights {
public:
    PackedWeights();
    ~PackedWeights();
    PackedWeights(const PackedWeights &) = delete;
    PackedWeights & operator=(const PackedWeights &) = delete;

    engine::core::TensorValue linear_bf16(engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input, const engine::modules::LinearWeights & weights);
    size_t bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ming_cpu::trial
