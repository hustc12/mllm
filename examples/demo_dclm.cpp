/**
 * @file demo_dclm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cmdline.h"
#include "models/dclm/configuration_dclm.hpp"
#include "models/dclm/modeling_dclm.hpp"
#include "models/dclm/tokenization_dclm.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/dclm_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/DCLM-1B-q4_0x4.mllm");
    cmdParser.add<string>("billion", 'b', "[1B]", false, "1B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = DCLMTokenizer(vocab_path);
    DCLMConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = DCLMTransformer(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications.",
    };

    auto processOutput = [&](std::string &text) -> std::pair<bool, std::string> {
        if (text == "<|im_start|>" || text == "<|im_end|>" || text == "<unk>") return {true, ""};
        if (text == "<|endoftext|>") return {false, ""};
        return {true, text};
    };

    auto addSystemPrompt = [](const std::string &text) -> std::string {
        std::string ret;
        std::string pre =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        ret = pre + text;
        std::string end = "<|im_end|>\n<|im_start|>assistant\n";
        ret = ret + end;
        return ret;
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = addSystemPrompt(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(input_str, i);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 100,
            .do_sample = true,
            .temperature = 0.3F,
            .top_k = 50,
            .top_p = 0.F,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = processOutput(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });
        std::cout << "\n";
    }
}