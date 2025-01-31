//
// Created by xwk on 24-10-23.
//
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"
#include "cmdline.h"
#include <vector>

/*
 * an intent to support gte-small BertModel to do text embedding
 * current implementation is just a very basic example with a simple WordPiece tokenizer and a simple BertModel
 * not support batch embedding
 * */

int main(int argc, char *argv[]) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/gte-small-fp32.mllm");
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/gte_vocab.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    string vocab_path = cmdParser.get<string>("vocab");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    BertTokenizer tokenizer(vocab_path, true);
    auto config = BertConfig();
    auto model = BertModel(config);
    model.load(model_path);

    string text1 = "Help me set an alarm at 21:30";
    string text2 = "Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text.";
    vector<string> texts = {text1, text2};
    for (auto &text : texts) {
        auto inputs = tokenizer.tokenizes(text);
        std::cout << "DEBUGGING - Input Length = " << inputs.at(0).size() << std::endl;
        auto start_time = mllm_time_us();
        auto res = model({inputs[0], inputs[1], inputs[2]})[0];
        auto end_time = mllm_time_us();
        std::cout << "DEBUGGING - Inference Time = " << (end_time - start_time) / 1000.0F << "ms" << std::endl;
        res.printData<float>();
    }

    return 0;
}
