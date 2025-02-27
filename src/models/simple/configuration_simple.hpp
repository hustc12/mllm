#ifndef CONFIGURATION_SIMPLE_HPP
#define CONFIGURATION_SIMPLE_HPP

#include "Layer.hpp"
#include "Types.hpp"

using namespace mllm;
using namespace std;

class SimpleNameConfig {
public:
    void init() {
        conv_name = "conv";
    }
    string conv_name;
};

struct SimpleConfig {
    explicit SimpleConfig() {
        in_channels = 3;
        out_channels = 64;
        kernel_size = 3;
        stride = 1;
        padding = SAME;
        bias = true;
        names_config.init();
    }

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    PaddingType padding;
    bool bias;
    SimpleNameConfig names_config;
};

#endif // CONFIGURATION_SIMPLE_HPP