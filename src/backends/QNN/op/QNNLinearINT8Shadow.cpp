
#include "QNNLinearINT8Shadow.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNLinearINT8Shadow::QNNLinearINT8Shadow(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
    weight_.setBackend(bn);
    weightScale_.setBackend(bn);
    outputScale_.setBackend(bn);
    inputScale_.setBackend(bn);

    shadowWeight_.setBackend(bn);
    shadowTransposeWeight_.setBackend(bn);
}

ErrorCode QNNLinearINT8Shadow::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    
    outputs[0]->reshape(inputs[2]->batch(), inputs[2]->head(), inputs[2]->sequence(), inputs[2]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinearINT8Shadow::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

        std::cout << "shadow add output buffers" << std::endl;

        inputs[0]->setBackend(qnnBackend_);
        inputs[0]->setDtype(MLLM_TYPE_F16);
        inputs[0]->alloc();

        qnnBackend_->pushOutputBuffers(inputs[0]->hostPtr<uint8_t>());

        inputs[1]->setBackend(qnnBackend_);
        inputs[1]->setDtype(MLLM_TYPE_I8);
        inputs[1]->alloc();

        qnnBackend_->pushOutputBuffers(inputs[1]->hostPtr<uint8_t>());


        inputs[2]->setBackend(qnnBackend_);
        inputs[2]->setDtype(MLLM_TYPE_F32);
        inputs[2]->alloc();

        qnnBackend_->pushOutputBuffers(inputs[2]->hostPtr<uint8_t>());
    

    return MLLM_NO_ERROR;
}

ErrorCode QNNLinearINT8Shadow::load(AbstructLoader &loader) {

    string opName = name();
    std::string wordToRemove = ".shadow";

    int pos = opName.find(wordToRemove);
    if (pos != -1) {
        opName.erase(pos, wordToRemove.length());
    }

    weight_.setName(opName + ".weight");
    weight_.reshape(1, 1, in_features_, out_features_);
    weight_.setDtype(MLLM_TYPE_I8);
    weight_.alloc();
    loader.load(&weight_);

    
    weightScale_.setName(opName + ".weight.scale");
    weightScale_.reshape(1, 1, 1, 1);
    weightScale_.setDtype(MLLM_TYPE_F32);
    weightScale_.alloc();
    loader.load(&weightScale_);

    outputScale_.setName(opName + ".output_scale");
    outputScale_.reshape(1, 1, 1, 1);
    outputScale_.setDtype(MLLM_TYPE_F32);
    outputScale_.alloc();
    loader.load(&outputScale_);

    inputScale_.setName(opName + ".input_scale");
    inputScale_.reshape(1, 1, 1, 1);
    inputScale_.setDtype(MLLM_TYPE_F32);
    inputScale_.alloc();
    loader.load(&inputScale_);


    
    shadowWeight_.setName(opName + ".shadow.weight");
    shadowWeight_.reshape(1, 1, in_features_, out_features_);
    shadowWeight_.setDtype(MLLM_TYPE_I8);
    shadowWeight_.alloc();

    memcpy(shadowWeight_.hostPtr<int8_t>(), weight_.hostPtr<int8_t>(), in_features_*out_features_);
    

    shadowTransposeWeight_.setName(opName + ".shadow.transpose.weight");
    shadowTransposeWeight_.reshape(1, 1, out_features_, in_features_);
    shadowTransposeWeight_.setDtype(MLLM_TYPE_I8);
    shadowTransposeWeight_.alloc();

    memcpy(shadowTransposeWeight_.hostPtr<int8_t>(), weight_.hostPtr<int8_t>(), in_features_*out_features_);


    weight_.free();

    return Op::load(loader);
}

ErrorCode QNNLinearINT8Shadow::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}


ErrorCode QNNLinearINT8Shadow::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto opName = name();

    // inputs[0] linear dequant input __fp16
    // inputs[1] linear quant output  int8_t 
    // inputs[2] res sum              float

    std::cout << opName << "shadow execution" << std::endl;

    memcpy(outputs[0]->hostPtr<float>(), inputs[2]->hostPtr<float>(), inputs[2]->batch()*inputs[2]->head()*inputs[2]->sequence()*inputs[2]->dimension()*sizeof(float));

    // // input outliers
    for (int i = 0; i < inputs[0]->batch(); i++) {

        for (int h = 0; h < inputs[0]->head(); h++) {

            for (int j = 0; j< inputs[0]->sequence(); j++) {

                for (int k = 0; k <inputs[0]->dimension(); k++) {

                    if (inputs[0]->dataAt<__fp16>(i, h, j, k) > inputScale_.dataAt<float>(0,0,0,0)) {

                        for (int w=0; w < shadowWeight_.dimension(); w++) {

                            if (!(inputs[1]->dataAt<int8_t>(i, h, j, k) <= -128 ||  inputs[1]->dataAt<int8_t>(i, h, j, k) >= 127))

                                outputs[0]->setDataAt<float>(i,h,j,w, inputs[0]->dataAt<__fp16>(i,h,j,k) * shadowWeight_.dataAt<int8_t>(0,0,k,w) * weightScale_.dataAt<float>(0,0,0,0) + outputs[0]->dataAt<float>(i,h,j,w));
                        }

                    }


                }

            }

        }
        
        

    }

    // output outliers
    for (int i = 0; i < inputs[1]->batch(); i++) {
        
        for (int h = 0; h < inputs[1]->head(); h++) {

            for (int j = 0; j< inputs[1]->sequence(); j++) {

                for (int k = 0; k <inputs[1]->dimension(); k++) {

                    if (inputs[1]->dataAt<int8_t>(i, 0, j, k) <= -128 ||  inputs[1]->dataAt<int8_t>(i, 0, j, k) >= 127) {

                        float sum = 0.0f;
                        for (int w=0; w < shadowWeight_.sequence(); w++) {
                            sum += inputs[0]->dataAt<__fp16>(i,h,j,w) * shadowWeight_.dataAt<int8_t>(0,0,w,k) * weightScale_.dataAt<float>(0,0,0,0);
                            
                            sum = sum - inputs[1]->dataAt<int8_t>(i, h, j, k) * outputScale_.dataAt<float>(0,0,0,0);

                        }

                        outputs[0]->setDataAt<float>(i,h,j,k, sum + outputs[0]->dataAt<float>(i,h,j,k));

                    }


                }

            }
        }

    }


    
    return MLLM_NO_ERROR;
}

} // namespace mllm
