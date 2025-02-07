#ifndef MODELING_VITNPU_HPP
#define MODELING_VITNPU_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_vit.hpp"
// #include "backends/qnn/QNNBackend.hpp"
// #include "backends/qnn/QNNGraph.hpp"
#include "hexagon_protos.h"

namespace mllm {

struct PrefetchConfig {
    static constexpr size_t PREFETCH_STRIDE = 4;
    static constexpr size_t L1_CACHE_LINE = 64;
    static constexpr size_t L2_CACHE_LINE = 128;
    static constexpr bool ENABLE_DOUBLE_BUFFERING = true;
    // Cache hierarchy configuration
    static constexpr size_t L1_CACHE_SIZE = 32 * 1024;     // 32KB L1 cache
    static constexpr size_t L2_CACHE_SIZE = 512 * 1024;    // 512KB L2 cache
    static constexpr size_t L1_PREFETCH_SIZE = 16 * 1024;  // 16KB prefetch window for L1
    static constexpr size_t L2_PREFETCH_SIZE = 128 * 1024; // 128KB prefetch window for L2
};

// Add prefetching helper functions
namespace {
// Add block-based prefetching for large tensors
void prefetchLargeTensor(void *ptr, size_t size) {
#ifdef __hexagon__
    const size_t l2_block = PrefetchConfig::L2_CACHE_LINE * 16; // 2KB blocks for L2
    const size_t l1_block = PrefetchConfig::L1_CACHE_LINE * 8;  // 1KB blocks for L1
    float *p = static_cast<float *>(ptr);

    // Prefetch to L2 in larger blocks
    for (size_t i = 0; i < size; i += l2_block) {
        size_t block_end = std::min(i + l2_block, size);
        for (size_t j = i; j < block_end; j += PrefetchConfig::L2_CACHE_LINE) {
            uint32_t config = (0x80 << 16) | (0x40 << 8) | 0x40;
            Q6_l2fetch_AR(p + j, config);
        }

        // Prefetch critical portion to L1
        size_t l1_end = std::min(i + l1_block, block_end);
        for (size_t j = i; j < l1_end; j += PrefetchConfig::L1_CACHE_LINE) {
            Q6_dcfetch_A(p + j);
        }
    }
#else
    // For non-Hexagon architectures, use larger stride prefetching
    const size_t block_size = 1024 * 4; // 4KB blocks
    char *p = static_cast<char *>(ptr);

    for (size_t i = 0; i < size; i += block_size) {
        size_t block_end = std::min(i + block_size, size);
        for (size_t j = i; j < block_end; j += PrefetchConfig::L1_CACHE_LINE) {
            __builtin_prefetch(p + j, 0, 3);
        }
    }
#endif
}

void prefetchMemory(void *ptr, size_t size) {
    // std::cout << "DEBUGGING - Prefetching memory" << std::endl;
#ifdef __hexagon__
    // For Hexagon DSP
    const size_t cacheLine = 128; // Hexagon cache line size
    float *p = static_cast<float *>(ptr);

    // Prefetch to L2 cache first
    for (size_t i = 0; i < size; i += cacheLine) {
        // Configure L2 prefetch parameters
        uint32_t height = 0x40; // 64 lines
        uint32_t width = 0x80;  // 128 bytes
        uint32_t config = (width << 16) | (height << 8) | 0x40;
        Q6_l2fetch_AR(p + i, config);
    }

    // Then prefetch to L1 cache
    for (size_t i = 0; i < size; i += cacheLine) {
        Q6_dcfetch_A(p + i);
    }
#elif defined(__ARM_ARCH__)
    // For ARM architectures
    const size_t cacheLine = 64; // typical ARM cache line size
    char *p = static_cast<char *>(ptr);

    for (size_t i = 0; i < size; i += cacheLine) {
        __builtin_prefetch(p + i, 0, 3);
    }
#else
    // For other architectures
    // (void)ptr;
    // (void)size;

    const size_t cacheLine = 128; // typical ARM cache line size
    char *p = static_cast<char *>(ptr);

    for (size_t i = 0; i < size; i += cacheLine) {
        __builtin_prefetch(p + i, 0, 3);
    }

#endif
}

// Hierarchical prefetching for different cache levels
void prefetchHierarchical(void *ptr, size_t size) {
#ifdef __hexagon__
    float *p = static_cast<float *>(ptr);

    // L2 cache prefetching - larger blocks
    const size_t l2_block = PrefetchConfig::L2_CACHE_LINE * 16; // 2KB blocks
    for (size_t i = 0; i < size; i += l2_block) {
        size_t block_end = std::min(i + l2_block, size);
        // Aggressive L2 prefetch
        for (size_t j = i; j < block_end; j += PrefetchConfig::L2_CACHE_LINE) {
            uint32_t config = (0x80 << 16) | (0x40 << 8) | 0x40;
            Q6_l2fetch_AR(p + j, config);
        }

        // L1 cache prefetching - smaller blocks for hot data
        size_t l1_window = std::min(block_end - i, PrefetchConfig::L1_PREFETCH_SIZE);
        for (size_t j = i; j < i + l1_window; j += PrefetchConfig::L1_CACHE_LINE) {
            Q6_dcfetch_A(p + j);
        }
    }
#else
    char *p = static_cast<char *>(ptr);

    // Simulate hierarchical prefetching on non-Hexagon architectures
    const size_t l2_block = 4096; // 4KB blocks for L2-like prefetching
    const size_t l1_block = 256;  // 256B blocks for L1-like prefetching

    for (size_t i = 0; i < size; i += l2_block) {
        // L2-like prefetching
        for (size_t j = i; j < std::min(i + l2_block, size); j += PrefetchConfig::L1_CACHE_LINE) {
            __builtin_prefetch(p + j, 0, 2); // Lower temporal locality for L2
        }

        // L1-like prefetching for hot data
        size_t l1_window = std::min(size - i, l1_block);
        for (size_t j = i; j < i + l1_window; j += PrefetchConfig::L1_CACHE_LINE) {
            __builtin_prefetch(p + j, 0, 3); // Higher temporal locality for L1
        }
    }
#endif
}

void prefetchTensorData(Tensor &tensor) {
    if (!tensor.hostPtr<void>()) {
        return;
    }
    size_t tensorSize = tensor.cntSize();
    void *dataPtr = tensor.hostPtr<void>();

    // Choose prefetch strategy based on tensor size
    if (tensorSize <= PrefetchConfig::L1_CACHE_SIZE) {
        // Small tensors - only L1 prefetch
        prefetchMemory(dataPtr, tensorSize);
    } else if (tensorSize <= PrefetchConfig::L2_CACHE_SIZE) {
        // Medium tensors - L1 + L2 prefetch
        prefetchHierarchical(dataPtr, tensorSize);
    } else {
        // Large tensors - block-based hierarchical prefetch
        prefetchLargeTensor(dataPtr, tensorSize);
    }
}
} // namespace

// Helper function for NPU optimization
namespace vit {
inline pair<int, int> closestFactors(int n) {
    int root = static_cast<int>(sqrt(n));
    for (int i = root; i > 0; --i) {
        if (n % i == 0) {
            return {i, n / i};
        }
    }
    return {1, n};
}
} // namespace vit

// NPU part for attention computation
class ViTAttentionNPUPart1 final : public Module {
    Layer pre_attn_view;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_view;
    Layer k_view;
    Layer v_view;
    Layer q_dequant;
    Layer k_dequant;
    Layer v_dequant;
    Layer v_transpose;

public:
    ViTAttentionNPUPart1() = default;
    ViTAttentionNPUPart1(const ViTConfig &config, const string &base_name) {
        int hidden_dim = config.hidden_dim;
        int head_size = config.head_size;
        int head_dim = hidden_dim / head_size;

        pre_attn_view = View(-1, 1, -1, head_size * head_dim, base_name + "attn_split_view"); // TODO: Do we need pre_attn_view?

        // Match original ViT naming
        q_proj = Linear(hidden_dim, hidden_dim, true, base_name + "attention.attention.query");
        k_proj = Linear(hidden_dim, hidden_dim, true, base_name + "attention.attention.key");
        v_proj = Linear(hidden_dim, hidden_dim, true, base_name + "attention.attention.value");

        q_view = View(-1, head_size, -1, head_dim, base_name + "query_view");
        k_view = View(-1, head_size, -1, head_dim, base_name + "key_view");
        v_view = View(-1, head_size, -1, head_dim, base_name + "value_view");

        q_dequant = Dequantize(true, base_name + "query.dequantize");
        k_dequant = Dequantize(true, base_name + "key.dequantize", false);
        v_dequant = Dequantize(true, base_name + "value.dequantize", false);

        v_transpose = Transpose({0, 2, 3, 1}, base_name + "value.transpose");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // auto x = pre_attn_view(inputs[0]);
        auto x = inputs[0];
        auto query_states = q_proj(x);
        auto key_states = k_proj(x);
        auto value_states = v_proj(x);

        // query_states = q_view(query_states);
        // key_states = k_view(key_states);
        // value_states = v_view(value_states);

        // TODO: Do we need dequantize?
        // query_states = q_dequant(query_states);
        // key_states = k_dequant(key_states);
        // value_states = v_dequant(value_states);

        // value_states = v_transpose(value_states);
        return {query_states, key_states, value_states};
    }
};

// CPU part for attention computation
class ViTAttentionCPU final : public Module {
    Softmax softmax;
    Layer o_quantize;

public:
    ViTAttentionCPU() = default;
    ViTAttentionCPU(const ViTConfig &config, const string &base_name) {
        softmax = Softmax(DIMENSION, true, base_name + "attention.softmax");
        o_quantize = Quantize(true, base_name + "attention.output.quantize");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        auto qk = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION));
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);

        // o = o_quantize(o);
        return {o};
    }
};

// NPU part for MLP
class ViTMLPNPU final : public Module {
    Layer norm;
    Layer up_proj;
    Layer act;
    Layer down_proj;
    Layer residual_add;

public:
    ViTMLPNPU() = default;
    ViTMLPNPU(const ViTConfig &config, const string &base_name) {
        int hidden_dim = config.hidden_dim;
        int ffn_hidden = config.ffn_hidden;

        norm = LayerNorm(hidden_dim, true, 1e-6, base_name + "layernorm_after");
        up_proj = Linear(hidden_dim, ffn_hidden, true, base_name + "intermediate.dense");
        act = GELU(base_name + "intermediate.intermediate.act");
        down_proj = Linear(ffn_hidden, hidden_dim, true, base_name + "output.dense");
        residual_add = Add(base_name + "mlp.residual");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto residual = inputs[1];

        x = norm(x);
        x = up_proj(x);
        x = act(x);
        x = down_proj(x);
        x = residual_add(x, residual);
        return {x};
    }
};

// Add this class before ViTModel_NPU
class ViTEmbeddingNPU final : public Module {
    Layer patch_embed;
    Parameter cls_token;
    Parameter pos_embed;

public:
    ViTEmbeddingNPU() = default;
    ViTEmbeddingNPU(const ViTConfig &config) {
        patch_embed = Convolution2D(3, config.hidden_dim,
                                    {config.patch, config.patch},
                                    {config.patch, config.patch},
                                    VALID, true, "vit.embeddings.patch_embeddings.projection");

        cls_token = Parameter(1, 1, 1, config.hidden_dim, "vit.embeddings.cls_token");
        pos_embed = Parameter(1, (config.img_hw * config.img_hw / (config.patch * config.patch)) + 1,
                              1, config.hidden_dim, "vit.embeddings.position_embeddings");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = patch_embed(inputs[0]);
        x = x.transpose({{SEQUENCE, DIMENSION}, {HEAD, SEQUENCE}});
        x = x.flatten(HEAD, SEQUENCE);

        auto cls = cls_token();
        cls = cls.expand(x.batch(), 1, 1, x.dimension());
        x = Tensor::cat({cls, x}, Chl::SEQUENCE);
        x = x + pos_embed();
        return {x};
    }
};

// Main ViT model with NPU support
class ViTModel_NPU final : public Module {
public:
    ViTModel_NPU() = default;
    explicit ViTModel_NPU(const ViTConfig &config) {
        this->config = config;
        embedding = ViTEmbeddingNPU(config);

        // Create encoder blocks
        for (int i = 0; i < config.block_num; i++) {
            string base_name = "vit.encoder.layer." + std::to_string(i) + ".";

            auto attn_npu = std::make_unique<ViTAttentionNPUPart1>(config, base_name);
            auto attn_cpu = std::make_unique<ViTAttentionCPU>(config, base_name);
            auto mlp = std::make_unique<ViTMLPNPU>(config, base_name);

            attn_npu->to(MLLM_CPU);
            attn_cpu->to(MLLM_CPU);
            mlp->to(MLLM_CPU);

            attention_npu_layers.push_back(std::move(attn_npu));
            attention_cpu_layers.push_back(std::move(attn_cpu));
            mlp_layers.push_back(std::move(mlp));
        }

        norm = LayerNorm(config.hidden_dim, true, 1e-6, "vit.encoder.layer.0.layernorm_before");
        lm_head = Linear(config.hidden_dim, config.class_size, false, "classifier");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs, args)[0];

        // Prefetch first layer's data if using QNN backend
        // if (!attention_npu_layers.empty()) {
        //     prefetchTensorData(x);
        // }

        for (size_t i = 0; i < attention_npu_layers.size(); i++) {
            // Prefetch next layer's data if available and using QNN backend
            if (i + 1 < attention_npu_layers.size() && i % 4 == 0) {
                auto next_residual = x;
                prefetchTensorData(next_residual);
            }

            auto residual = x;

            // x = Tensor::toQNN({x})[0];
            auto qkv = attention_npu_layers[i]->Forward({x}, {});

            // qkv = Tensor::toCPU(qkv);
            auto attn_output = attention_cpu_layers[i]->Forward(qkv, {})[0];

            // auto npu_tensors = Tensor::toQNN({attn_output, residual});
            auto npu_tensors = Tensor::toCPU({attn_output, residual});
            x = mlp_layers[i]->Forward({npu_tensors[0], npu_tensors[1]}, {})[0];

            // Prefetch MLP layer data if available and using QNN backend
            if (i + 1 < mlp_layers.size() && i % 4 == 0) {
                prefetchTensorData(x);
            }
        }

        // x = Tensor::toCPU({x})[0];

        x = x.clip({}, {}, {0}, {});
        x = norm(x);
        x = lm_head(x);

        return {x};
    }

private:
    ViTEmbeddingNPU embedding;
    vector<unique_ptr<ViTAttentionNPUPart1>> attention_npu_layers;
    vector<unique_ptr<ViTAttentionCPU>> attention_cpu_layers;
    vector<unique_ptr<ViTMLPNPU>> mlp_layers;
    Layer norm;
    Layer lm_head;
    ViTConfig config;
};

} // namespace mllm

#endif // MODELING_VITNPU_HPP
