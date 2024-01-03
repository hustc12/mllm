
#include "CPUMul.hpp"

namespace mllm {

CPUMul::CPUMul(Backend *bn,  string opName, bool multiThread) :
    Op(bn, opName) {
}

ErrorCode CPUMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUMul  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
    CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMul::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUMul()" << std::endl;
    int N = inputs[0]->batch();
    int C = inputs[0]->head();
    int H = inputs[0]->sequence();
    int W = inputs[0]->dimension();

    if(inputs[0]->masterTensor() == nullptr && inputs[1]->masterTensor() == nullptr && inputs[0]->ctype() == inputs[1]->ctype()) {
        auto copy_size = N * C * H * W;
        auto in0_ptr = inputs[0]->hostPtr<float>();
        auto in1_ptr = inputs[1]->hostPtr<float>();
        auto out_ptr = outputs[0]->hostPtr<float>();
#pragma omp parallel for num_threads(4)
        for (int is = 0; is < copy_size; ++is) {
            out_ptr[is] = in0_ptr[is] * in1_ptr[is];
        }
    }else {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(4)
                    for (int w = 0; w < W; ++w) {
                        outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n, c, h, w) * inputs[1]->dataAt<float>(n, c, h, w));
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm
