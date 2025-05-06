#ifndef MODELING_SIMPLE_HPP
#define MODELING_SIMPLE_HPP

#include "Layer.hpp"
#include "Types.hpp"
#include "configuration_simple_conv.hpp"
#include <vector>

using namespace mllm;

class SimpleModel final : public Module {
public:
    SimpleModel() = default;

    SimpleModel(SimpleConfig &config) {
        conv1 = Convolution2D(
            config.in_channels,
            config.out_channels,
            {config.kernel_size, config.kernel_size},
            {config.stride, config.stride},
            config.padding,
            config.bias,
            config.names_config.conv1_name);
        conv2 = Convolution2D(
            config.out_channels,
            config.out_channels,
            {config.kernel_size, config.kernel_size},
            {config.stride, config.stride},
            config.padding,
            config.bias,
            config.names_config.conv2_name);
        conv3 = Convolution2D(
            config.out_channels,
            config.out_channels,
            {config.kernel_size, config.kernel_size},
            {config.stride, config.stride},
            config.padding,
            config.bias,
            config.names_config.conv3_name);
        conv4 = Convolution2D(
            config.out_channels,
            config.out_channels,
            {config.kernel_size, config.kernel_size},
            {config.stride, config.stride},
            config.padding,
            config.bias,
            config.names_config.conv4_name);
        relu = ReLU(config.names_config.relu_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = conv1(inputs[0]);
        x = relu(x);
        x = conv2(x);
        x = relu(x);
        x = conv3(x);
        x = relu(x);
        x = conv4(x);
        return {x};
    }

private:
    Layer conv1;
    Layer conv2;
    Layer conv3;
    Layer conv4;
    Layer relu;
};

#endif // MODELING_SIMPLE_HPP