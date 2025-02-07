//
// Created by Rongjie Yi on 2024/03/07 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/tinyllama/modeling_tinyllama_dsp.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/tinyllama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/tinyllama-1.1b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);
    string system_prompt_start = " You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.<|USER|>";
    string system_prompt_end = "<|ASSISTANT|>";
    tokenizer.set_chat_template(system_prompt_start, system_prompt_end);

    TinyLLaMAConfig config(tokens_limit, "1.5B", HFHUBROPE);
    auto model = TinyLLaMAModel_DSP(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "Please give me a short introduce of the TinyLLaMA model."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = tokenizer.apply_chat_template(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << "DEBUGGING: input_tensor size: " << input_tensor.size() << std::endl;
        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        auto start_time = mllm_time_us();
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        auto end_time = mllm_time_us();
        std::cout << "\nDEBUGGING - Inference Time = " << (end_time - start_time) / 1000.0F << " ms" << std::endl;
        printf("\n");
        model.profiling();
    }

    return 0;
}