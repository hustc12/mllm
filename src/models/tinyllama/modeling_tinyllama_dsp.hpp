//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef MODELING_TINYLLAMA_HPP
#define MODELING_TINYLLAMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_tinyllama.hpp"
#include "models/llama/modeling_llama.hpp"
#include "hexagon_protos.h"

using namespace mllm;

// Add prefetching configuration and helpers
namespace {
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

// Prefetching helper functions (same as in ViT)
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
    const size_t cacheLine = 128; // typical cache line size
    char *p = static_cast<char *>(ptr);

    for (size_t i = 0; i < size; i += cacheLine) {
        __builtin_prefetch(p + i, 0, 3);
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
        // std::cout << "DEBUGGING: prefetching small tensor - Tensor Size =  " << tensorSize / 1024 << " KB" << std::endl;
        // Small tensors - only L1 prefetch
        prefetchMemory(dataPtr, tensorSize);
    } else if (tensorSize <= PrefetchConfig::L2_CACHE_SIZE) {
        // std::cout << "DEBUGGING: prefetching medium tensor - Tensor Size =  " << tensorSize / 1024 << " KB" << std::endl;
        // Medium tensors - L1 + L2 prefetch
        prefetchHierarchical(dataPtr, tensorSize);
    } else {
        // std::cout << "DEBUGGING: prefetching large tensor - Tensor Size =  " << tensorSize / 1024 << " KB" << std::endl;
        // Large tensors - block-based hierarchical prefetch
        prefetchLargeTensor(dataPtr, tensorSize);
    }
}
} // namespace

class TinyLLaMABlock final : public Module {
    MultiHeadAttention attention;
    LLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    TinyLLaMABlock() = default;
    TinyLLaMABlock(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, const LLaMANameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size, hidden_dim / head_size, SPLIT_NONE, false, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, false, names, base_name + names._attn_base_name);
        mlp = LLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // Prefetch attention inputs
        prefetchTensorData(inputs[0]);

        auto x = norm1(inputs[0]);

        // Prefetch MLP inputs while attention is running
        auto mlp_input = x + inputs[0];
        prefetchTensorData(mlp_input);

        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];

        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
};

class TinyLLaMAModel_DSP final : public Module {
    Layer embedding;
    vector<TinyLLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit TinyLLaMAModel_DSP(const TinyLLaMAConfig &config) :
        TinyLLaMAModel_DSP(config.vocab_size, config.hidden_dim, config.head_size, config.kv_head_size, config.ffn_hidden, config.block_num,
                           config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                           config.names_config, config.names_config.blk_name) {
    }
    TinyLLaMAModel_DSP(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num,
                       RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
                       const LLaMANameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<TinyLLaMABlock>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // Prefetch first block's data
        if (!blocks.empty()) {
            prefetchTensorData(x);
            if (blocks.size() > 1) {
                // Also prefetch next block's input
                auto next_x = x;
                prefetchTensorData(next_x);
            }
        }

        // Process blocks with prefetching
        for (size_t i = 0; i < blocks.size(); i++) {
            // Prefetch next block's data if available
            if (i + 1 < blocks.size() && i % PrefetchConfig::PREFETCH_STRIDE == 0) {
                auto next_input = x;
                prefetchTensorData(next_input);
            }

            x = blocks[i]({x})[0];
        }

        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

#endif // MODELING_TINYLLAMA_HPP
