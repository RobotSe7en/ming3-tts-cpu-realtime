#pragma once

#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/core/execution_context.h"

namespace ming_cpu {

// Fuses the checkpoint's Linear -> BF16-round sequence. Large matrices and
// unsupported storage types use the ordinary GGML implementation.
engine::core::TensorValue linear_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::LinearWeights & weights, bool allow_avx2 = true);

engine::core::TensorValue rms_norm_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input, const engine::modules::NormWeights & weights);
engine::core::TensorValue add_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs, const engine::core::TensorValue & rhs);

std::vector<float> make_gelu_bf16_table(engine::core::ExecutionContext & execution);
// Input must already be rounded to BF16. The table must outlive the graph.
engine::core::TensorValue gelu_bf16(engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input, const std::vector<float> & table);

}  // namespace ming_cpu
