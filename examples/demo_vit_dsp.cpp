#include <iostream>
#include <vector>
#include "cmdline.h"
#include "models/vit/modeling_vit_npu.hpp"
#include "models/vit/labels_vit.hpp"
#include "models/vit/processing_vit.hpp"
#include "backends/cpu/CPUBackend.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/vit-base-patch16-224-int8.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // Initialize ViT processor for image preprocessing
    auto processor = ViTProcessor();

    // Initialize ViT model with NPU support
    ViTConfig config("base", 16, 224, imagenet_id2label.size());
    auto model = ViTModel_NPU(config);
    model.load(model_path);

    // Test images
    vector<string> imgs = {
        "../assets/cat.jpg",
        "../assets/dog_image.jpg",
        "../assets/bird_image.jpg",
        "../assets/car_image.jpg",
        "../assets/bus.png"};

    // Process each image
    for (auto &img : imgs) {
        try {
            // Preprocess image
            auto input_tensor = processor.process(img, 224);

            // Create unique graph name for this inference
            std::string graph_name = "vit_graph_" + std::to_string(rand());

            // Run inference on NPU
            auto result = model({input_tensor}, graph_name);

            // Post-process results
            auto token_idx = processor.postProcess(result[0]);

            // Print results
            std::cout << "Image: " << img << std::endl;
            std::cout << "Predicted class: " << imagenet_id2label[token_idx] << std::endl;
            std::cout << "-------------------" << std::endl;

            // // Free graph resources after inference
            // model.free(graph_name);
        } catch (const std::exception &e) {
            std::cerr << "Error processing image " << img << ": " << e.what() << std::endl;
            continue;
        }
    }

    // // Cleanup
    // model.allFree();

    return 0;
}
