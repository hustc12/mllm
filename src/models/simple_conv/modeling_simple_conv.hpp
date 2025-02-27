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
        conv = Convolution2D(
            config.in_channels,
            config.out_channels,
            {config.kernel_size, config.kernel_size},
            {config.stride, config.stride},
            config.padding,
            config.bias,
            config.names_config.conv_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = conv(inputs[0]);
        return {x};
    }

private:
    Layer conv;
};

#endif // MODELING_SIMPLE_HPP