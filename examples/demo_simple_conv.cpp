#include "models/simple_conv/configuration_simple_conv.hpp"
#include "models/simple_conv/modeling_simple_conv.hpp"
#include "models/simple_conv/processing_simple_conv.hpp"
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
    auto processor = SimpleConvProcessor();

    auto config = SimpleConfig();
    auto model = SimpleModel(config);
    model.load(model_path);

    // Create a dummy input tensor (1, 3, 32, 32)
    // Tensor input(1, 3, 32, 32, Backend::global_backends[MLLM_CPU], true);
    // vector<int> shape = {1, 3, 32, 32};
    // Tensor input(shape);
    // input.setName("input");
    // input.alloc();
    // input.fullData(1.0f);

    string imgs = "../assets/cat.jpg";
    auto input_tensor = processor.process(imgs, 224);
    // input_tensor.printData<float>();

    // // Run inference
    auto start_time = mllm_time_us();
    auto output = model({input_tensor})[0];
    auto end_time = mllm_time_us();
    std::cout << "Inference Time = " << (end_time - start_time) / 1000.0F << "ms" << std::endl;
    // output.printData<float>();

    return 0;
}