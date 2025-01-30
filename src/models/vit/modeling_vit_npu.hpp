#ifndef MODELING_VITNPU_HPP
#define MODELING_VITNPU_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_vit.hpp"

namespace mllm {

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
    int hidden_dim;
    int num_heads;
    int head_dim;

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
        hidden_dim = config.hidden_dim;
        num_heads = config.head_size;
        head_dim = config.hidden_dim / num_heads;

        pre_attn_view = View(-1, 1, -1, num_heads * head_dim, base_name + "attn_split_view");

        q_proj = Linear(hidden_dim, hidden_dim, true, base_name + "query");
        k_proj = Linear(hidden_dim, hidden_dim, true, base_name + "key");
        v_proj = Linear(hidden_dim, hidden_dim, true, base_name + "value");

        q_view = View(-1, num_heads, -1, head_dim, base_name + "query_view");
        k_view = View(-1, num_heads, -1, head_dim, base_name + "key_view");
        v_view = View(-1, num_heads, -1, head_dim, base_name + "value_view");

        q_dequant = Dequantize(true, base_name + "query.dequantize");
        k_dequant = Dequantize(true, base_name + "key.dequantize", false);
        v_dequant = Dequantize(true, base_name + "value.dequantize", false);

        v_transpose = Transpose({0, 2, 3, 1}, base_name + "value.transpose");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = pre_attn_view(inputs[0]);

        auto query_states = q_proj(x);
        auto key_states = k_proj(x);
        auto value_states = v_proj(x);

        query_states = q_view(query_states);
        key_states = k_view(key_states);
        value_states = v_view(value_states);

        query_states = q_dequant(query_states);
        key_states = k_dequant(key_states);
        value_states = v_dequant(value_states);

        value_states = v_transpose(value_states);
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
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
        o_quantize = Quantize(true, base_name + "proj.quantize");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        auto qk = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION));
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);

        o = o_quantize(o);
        return {o};
    }
};

// NPU part for MLP and output projection
class ViTMLPNPU final : public Module {
    Layer pre_mlp_view;
    Layer mlp_fc1;
    Layer mlp_act;
    Layer mlp_fc2;
    Layer post_mlp_view;
    Layer layernorm;
    Layer out_proj;
    Layer post_attn_add;
    Layer post_mlp_add;

public:
    ViTMLPNPU() = default;
    ViTMLPNPU(const ViTConfig &config, const string &base_name) {
        int hidden_dim = config.hidden_dim;
        int intermediate_size = config.hidden_dim * 4; // MLP typically uses 4x hidden size

        pre_mlp_view = View(1, vit::closestFactors(config.img_hw).first,
                            vit::closestFactors(config.img_hw).second, hidden_dim,
                            base_name + "mlp_view1");

        mlp_fc1 = Linear(hidden_dim, intermediate_size, true, base_name + "mlp.fc1");
        mlp_act = GELU(base_name + "mlp.act");
        mlp_fc2 = Linear(intermediate_size, hidden_dim, true, base_name + "mlp.fc2");

        post_mlp_view = View(1, 1, -1, hidden_dim, base_name + "mlp_view2");

        layernorm = LayerNorm(hidden_dim, true, 1e-6, base_name + "layernorm");
        out_proj = Linear(hidden_dim, hidden_dim, true, base_name + "proj");

        post_attn_add = Add(base_name + "attn_residual");
        post_mlp_add = Add(base_name + "mlp_residual");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto attn_output = inputs[0];
        auto residual = inputs[1];

        auto x = post_attn_add(attn_output, residual);
        x = layernorm(x);

        x = pre_mlp_view(x);
        x = mlp_fc1(x);
        x = mlp_act(x);
        x = mlp_fc2(x);
        x = post_mlp_view(x);

        x = post_mlp_add(x, residual);
        return {x};
    }
};

// Main ViT model with NPU support
class ViTModel_NPU final : public Module {
public:
    ViTModel_NPU() = default;
    ViTModel_NPU(const ViTConfig &config) {
        this->config = config; // Store config for later use

        patch_embed = Linear(config.patch * config.patch * 3, config.hidden_dim, true, "patch_embed");

        cls_token = Parameter(1, 1, 1, config.hidden_dim, "cls_token");
        pos_embed = Parameter(1, 1, (config.img_hw * config.img_hw / (config.patch * config.patch)) + 1,
                              config.hidden_dim, "pos_embed");

        // Create encoder blocks
        for (int i = 0; i < config.ffn_hidden; i++) {
            string base_name = "encoder.layer." + std::to_string(i) + ".";

            auto attn_npu = std::make_unique<ViTAttentionNPUPart1>(config, base_name);
            auto attn_cpu = std::make_unique<ViTAttentionCPU>(config, base_name);
            auto mlp = std::make_unique<ViTMLPNPU>(config, base_name);

            attn_npu->to(MLLM_QNN);
            attn_cpu->to(MLLM_CPU);
            mlp->to(MLLM_QNN);

            attention_npu_layers.push_back(std::move(attn_npu));
            attention_cpu_layers.push_back(std::move(attn_cpu));
            mlp_layers.push_back(std::move(mlp));
        }

        layernorm = LayerNorm(config.hidden_dim, true, 1e-6, "layernorm");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // Reshape input for patch embedding
        auto B = inputs[0].batch();
        auto H = config.img_hw;
        auto W = config.img_hw;
        auto P = config.patch;

        // Reshape and patch embedding
        inputs[0].reshape(B, 1, (H / P) * (W / P), P * P * 3);
        auto x = patch_embed(inputs[0]);

        // Add cls token and position embeddings
        auto cls = cls_token();
        cls = cls.expand(x.batch(), 1, 1, x.dimension());

        // Use Tensor::cat with proper Chl enum
        x = Tensor::cat({cls, x}, Chl::SEQUENCE);
        x = x + pos_embed();

        // Process through encoder blocks
        for (size_t i = 0; i < attention_npu_layers.size(); i++) {
            auto residual = x;

            // NPU attention part 1
            x = Tensor::toQNN({x})[0];
            auto qkv = attention_npu_layers[i]->Forward({x}, {});

            // CPU attention part
            qkv = Tensor::toCPU(qkv);
            auto attn_output = attention_cpu_layers[i]->Forward(qkv, {})[0];

            // NPU MLP part
            auto npu_tensors = Tensor::toQNN({attn_output, residual});
            x = mlp_layers[i]->Forward({npu_tensors[0], npu_tensors[1]}, {})[0];
            x = Tensor::toCPU({x})[0];
        }

        x = layernorm(x);
        return {x};
    }

private:
    Layer patch_embed;
    Parameter cls_token;
    Parameter pos_embed;
    vector<unique_ptr<ViTAttentionNPUPart1>> attention_npu_layers;
    vector<unique_ptr<ViTAttentionCPU>> attention_cpu_layers;
    vector<unique_ptr<ViTMLPNPU>> mlp_layers;
    Layer layernorm;
    ViTConfig config; // Store config for use in Forward
};

} // namespace mllm

#endif // MODELING_VITNPU_HPP