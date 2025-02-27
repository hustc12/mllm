#include "models/simple/configuration_simple.hpp"
#include "models/simple/modeling_simple.hpp"
#include "cmdline.h"
#include <vector>

int main(int argc, char *argv[]) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/simple_conv_model.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // Create config and model
    auto config = SimpleConfig();
    auto model = SimpleModel(config);
    model.load(model_path);

    // Create a dummy input tensor (1, 3, 32, 32)
    auto input = Tensor::randn({1, 3, 32, 32});

    // Run inference
    auto output = model({input})[0];
    output.printData<float>();

    return 0;
}