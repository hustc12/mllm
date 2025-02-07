#ifndef MODELING_YOLOX_X_HPP
#define MODELING_YOLOX_X_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_yolox_x.hpp"

using namespace mllm;

// CSPLayer implementation
class CSPLayer final : public Module {
    Layer conv1;
    Layer conv2;
    Layer conv3;
    vector<Layer> m;

public:
    CSPLayer() = default;
    CSPLayer(int in_channels, int out_channels, int num_blocks, bool first_layer, const string &base_name) {
        conv1 = Convolution2D(in_channels, out_channels, {1, 1}, {1, 1}, SAME, true, base_name + "conv1.conv");
        conv2 = Convolution2D(in_channels, out_channels, {1, 1}, {1, 1}, SAME, true, base_name + "conv2.conv");
        conv3 = Convolution2D(2 * out_channels, out_channels, {1, 1}, {1, 1}, SAME, true, base_name + "conv3.conv");

        // Initialize bottleneck layers
        for (int i = 0; i < num_blocks; i++) {
            string block_name = base_name + "m." + std::to_string(i) + ".";
            Layer conv1 = Convolution2D(out_channels, out_channels, {1, 1}, {1, 1}, SAME, true, block_name + "conv1.conv");
            Layer conv2 = Convolution2D(out_channels, out_channels, {3, 3}, {1, 1}, SAME, true, block_name + "conv2.conv");
            m.push_back(conv1);
            m.push_back(conv2);
        }
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto x1 = conv1(x);
        auto x2 = conv2(x);

        for (size_t i = 0; i < m.size(); i += 2) {
            auto m_x = m[i](x1);
            m_x = m[i + 1](m_x);
            x1 = x1 + m_x;
        }

        x = Tensor::cat({x1, x2}, CHANNLE);
        x = conv3(x);
        return {x};
    }
};

// Focus module implementation
class Focus final : public Module {
    Layer conv;

public:
    Focus() = default;
    Focus(int in_channels, int out_channels, const string &base_name) {
        conv = Convolution2D(in_channels * 4, out_channels, {3, 3}, {1, 1}, SAME, true, base_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        // Implement focus operation
        x = conv(x);
        return {x};
    }
};

// YOLOX Head implementation
class YOLOXHead final : public Module {
    vector<Layer> stems;
    vector<Layer> cls_convs;
    vector<Layer> reg_convs;
    vector<Layer> cls_preds;
    vector<Layer> reg_preds;
    vector<Layer> obj_preds;

public:
    YOLOXHead() = default;
    YOLOXHead(const YOLOXConfig &config, const string &base_name) {
        for (size_t i = 0; i < config.strides.size(); i++) {
            string stem_name = base_name + "stems." + std::to_string(i) + ".";
            stems.push_back(Convolution2D(config.in_channels[i], config.act_channels[i], {1, 1}, {1, 1}, SAME, true, stem_name));

            string cls_name = base_name + "cls_convs." + std::to_string(i) + ".";
            cls_convs.push_back(Convolution2D(config.act_channels[i], config.act_channels[i], {3, 3}, {1, 1}, SAME, true, cls_name));

            string reg_name = base_name + "reg_convs." + std::to_string(i) + ".";
            reg_convs.push_back(Convolution2D(config.act_channels[i], config.act_channels[i], {3, 3}, {1, 1}, SAME, true, reg_name));

            cls_preds.push_back(Linear(config.act_channels[i], config.num_classes, true, base_name + "cls_preds." + std::to_string(i)));
            reg_preds.push_back(Linear(config.act_channels[i], 4, true, base_name + "reg_preds." + std::to_string(i)));
            obj_preds.push_back(Linear(config.act_channels[i], 1, true, base_name + "obj_preds." + std::to_string(i)));
        }
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<Tensor> outputs;
        for (size_t i = 0; i < inputs.size(); i++) {
            auto x = stems[i](inputs[i]);

            auto cls_feat = cls_convs[i](x);
            auto cls_output = cls_preds[i](cls_feat);

            auto reg_feat = reg_convs[i](x);
            auto reg_output = reg_preds[i](reg_feat);
            auto obj_output = obj_preds[i](reg_feat);

            outputs.push_back(Tensor::cat({reg_output, obj_output, cls_output}, CHANNLE));
        }
        return outputs;
    }
};

// Main YOLOX model implementation
class YOLOXModel final : public Module {
    Focus backbone_focus;
    vector<CSPLayer> backbone_layers;
    YOLOXHead head;

public:
    explicit YOLOXModel(const YOLOXConfig &config) {
        // Initialize backbone
        backbone_focus = Focus(config.channels, config.stem_out_channels, config.names_config.backbone_stem_name);

        // Initialize CSP layers
        backbone_layers.push_back(CSPLayer(config.stem_out_channels, config.dark2_out_channels, 3, true,
                                           config.names_config.backbone_dark_name + "2."));
        backbone_layers.push_back(CSPLayer(config.dark2_out_channels, config.dark3_out_channels, 9, false,
                                           config.names_config.backbone_dark_name + "3."));
        backbone_layers.push_back(CSPLayer(config.dark3_out_channels, config.dark4_out_channels, 9, false,
                                           config.names_config.backbone_dark_name + "4."));
        backbone_layers.push_back(CSPLayer(config.dark4_out_channels, config.dark5_out_channels, 3, false,
                                           config.names_config.backbone_dark_name + "5."));

        // Initialize head
        head = YOLOXHead(config, config.names_config.head_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // Wrap input tensor in vector for Focus operator
        auto x = backbone_focus({inputs[0]})[0];

        vector<Tensor> features;
        for (auto &layer : backbone_layers) {
            // Create vector of inputs for layer
            vector<Tensor> layer_inputs = {x};
            // Get layer output and store first tensor
            auto layer_output = layer(layer_inputs)[0];
            features.push_back(layer_output);
            // Update x for next iteration
            x = layer_output;
        }

        return head(features, args);
    }
};

#endif // MODELING_YOLOX_X_HPP
