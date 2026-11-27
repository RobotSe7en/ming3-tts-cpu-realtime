// Compare the CPU fusion with audio.cpp's original rounded RoPE graph.
#include TTS_MING_OPT_SOURCE
#include <omp.h>
#include <iostream>

int main() try {
    using namespace engine;
    core::BackendConfig backend;
    backend.type = core::BackendType::Cpu;
    backend.threads = 4;
    omp_set_num_threads(4);
    core::ExecutionContext execution(backend);
    int cases = 0;
    for (int64_t batch : {1, 2, 7}) for (int64_t steps : {2, 3}) for (int64_t heads : {2, 16}) {
        const int64_t dim = 64, hidden = heads * dim;
        auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext build{ctx, "rope.test", core::BackendType::Cpu};
        auto input = core::make_tensor(build, GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, steps, hidden}));
        auto positions = core::make_tensor(build, GGML_TYPE_I32,
            core::TensorShape::from_dims({steps}));
        auto factors = core::make_tensor(build, GGML_TYPE_F32,
            core::TensorShape::from_dims({dim / 2}));
        for (auto * tensor : {input.tensor, positions.tensor, factors.tensor}) {
            ggml_set_input(tensor);
            ggml_set_output(tensor);
        }
        auto * pairs = ggml_reshape_4d(ctx, input.tensor, 2, dim / 2, heads, batch * steps);
        pairs = ggml_cont(ctx, ggml_permute(ctx, pairs, 1, 0, 2, 3));
        auto reference = core::wrap_tensor(ggml_reshape_4d(ctx, pairs, dim, heads, steps, batch),
            core::TensorShape::from_dims({batch, steps, heads, dim}), GGML_TYPE_F32);
        modules::RoPEConfig rope{dim, GGML_ROPE_TYPE_NEOX, 10000.0f};
        rope.intermediate_round_type = GGML_TYPE_BF16;
        reference = modules::RoPEModule(rope).build(build, reference, positions, &factors);
        reference = modules::Bf16RoundModule{}.build(build, reference);
        pairs = ggml_reshape_4d(ctx, reference.tensor, dim / 2, 2, heads, batch * steps);
        pairs = ggml_cont(ctx, ggml_permute(ctx, pairs, 1, 0, 2, 3));
        auto * expected = ggml_reshape_3d(ctx, pairs, hidden, steps, batch);

        auto * basis = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim / 2, 1, steps, 1);
        auto * identity = ggml_concat(ctx, ggml_fill(ctx, basis, 1.0f), ggml_fill(ctx, basis, 0.0f), 0);
        auto * table = ggml_round_bf16(ctx, ggml_rope_ext(ctx, identity, positions.tensor,
            factors.tensor, dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
        table = ggml_reshape_2d(ctx, table, dim, steps);
        auto * actual = ggml_map_custom2(ctx, input.tensor, table,
            modules::rounded_rope_cpu, GGML_N_TASKS_MAX, nullptr);
        ggml_set_output(expected);
        ggml_set_output(actual);
        auto * graph = ggml_new_graph_custom(ctx, 2048, false);
        ggml_build_forward_expand(graph, expected);
        ggml_build_forward_expand(graph, actual);
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("Test allocation failed");
        std::vector<float> values(batch * steps * hidden), frequency(dim / 2);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(i * 0.173f) * 3.25f;
        for (size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.0f + (i % 7) * 0.001f;
        core::write_tensor_f32(input, values);
        core::write_tensor_f32(factors, frequency);
        const std::array<int32_t, 3> ids{0, 1, 2};
        ggml_backend_tensor_set(positions.tensor, ids.data(), 0, steps * sizeof(int32_t));
        for (int repeat = 0; repeat < 3; ++repeat) {
            if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Test graph failed");
            }
            const auto lhs = core::read_tensor_f32(expected);
            const auto rhs = core::read_tensor_f32(actual);
            if (std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) != 0) {
                for (size_t i = 0; i < lhs.size(); ++i) if (lhs[i] != rhs[i]) {
                    std::cerr << "Mismatch batch=" << batch << " steps=" << steps << " heads=" << heads
                              << " index=" << i << " expected=" << lhs[i] << " actual=" << rhs[i] << '\n';
                    break;
                }
                throw std::runtime_error("Rounded RoPE differs from reference");
            }
        }
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        ++cases;
    }
    std::cout << cases << " RoPE shapes passed bitwise comparison across repeated executions\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
