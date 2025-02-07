#include "QNNGraph.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include <cstring>
#include <memory>
#ifdef DEBUGPRINT
#include "Timing.hpp"
#endif

#include "QNNBackend.hpp"

namespace mllm {

QNNGraph::QNNGraph(const NetParameter &param, Backend *bn,
                   unordered_map<string, shared_ptr<Tensor>> &external_tensors,
                   int threadCount) :
    Graph(param, bn, external_tensors, threadCount) {

}


void QNNGraph::setUpTensors(std::string name) {

    // change to use merge op output as graph input tensor
    vector<shared_ptr<Tensor>> graph_in_tensors;
    if (ops_[op_names_[0]]->type() == SPLITINPUT) {
        graph_in_tensors = ops_output_tensors_[op_names_[0]];
    } else {
        graph_in_tensors = ops_input_tensors_[op_names_[0]];
    }
    
    // set graph out tensor TensorType
    auto &graph_out_tensors = ops_output_tensors_[op_names_[op_names_.size() - 1]];
    for (auto &t : graph_out_tensors) {
        t->setTtype(GRAPH_OUTPUT);
        t->alloc();
    }

    this->backend_->onSetUpStart(graph_in_tensors, graph_out_tensors, name);

    // set up tensors of ops
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
            ops_[op_name]->setUp(ops_input_tensors_[op_name],
                                 ops_output_tensors_[op_name]);
        } else {
            // std::cout << "op_name:" << op_name << " is not do" << std::endl;
        }
    }

    this->backend_->onSetUpEnd(graph_in_tensors, graph_out_tensors, name);
}

// WARNING: non virtual override function, all features should be merged into the origin function
const vector<shared_ptr<Tensor>> &QNNGraph::forward(std::string graphName) {
    for (size_t i = 0; i < op_names_.size(); i++) {
        std::cout << "DEBUGGING - Forwarding operation: " << op_names_[i] << std::endl;
        const auto &op_name = op_names_[i];

        // Layer-aware prefetching for next operation
        if (i + 1 < op_names_.size()) {
            const auto &next_op_name = op_names_[i + 1];
            if (ops_not_inputs_empty_[next_op_name]) {
                auto &next_inputs = ops_input_tensors_[next_op_name];

                // Prefetch input tensors for next operation
                for (auto &next_tensor : next_inputs) {
                    std::cout << "DEBUGGING - Prefetching tensor: " << next_tensor->name() << std::endl;
                    if (next_tensor) {
                        std::cout << "DEBUGGING - Tensor is not null" << std::endl;
                        auto *qnn_backend = dynamic_cast<QNNBackend *>(this->backend_);
                        if (qnn_backend) {
                            qnn_backend->prefetchTensorData(next_tensor);
                        }
                    } else {
                        std::cout << "DEBUGGING - Tensor is null" << std::endl;
                    }
                }
            }
        }

        // Execute current operation
        if (ops_not_inputs_empty_[op_name]) {
#ifdef SAVECHECK
            for (auto &t : ops_input_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif
#ifdef DEBUGPRINT
            uint64_t t_start = mllm_time_us();
#endif
            if (ops_[op_name]->type() == LINEARINT8SHADOW || ops_[op_name]->type() == ROPE)
                continue;
            ops_[op_name]->execute(ops_input_tensors_[op_name],
                                   ops_output_tensors_[op_name]);

#ifdef SAVECHECK
            for (auto &t : ops_output_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif

#ifdef DEBUGPRINT
            uint64_t t_end = mllm_time_us();
            std::cout << "" << op_name
                      << "       exe_time:" << (t_end - t_start) / 1000.0F << " ms"
                      << std::endl;
#endif
        } else {
            //            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }

    this->backend_->onExecuteStart(ops_input_tensors_[op_names_[0]], ops_output_tensors_[op_names_[op_names_.size() - 1]], graphName);

    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

void QNNGraph::free(std::string graphName) {
    auto *qnn_backend = dynamic_cast<QNNBackend *>(this->backend_);
    qnn_backend->freeGraphDataStructure(graphName);
}

void QNNGraph::allFree() {
    auto *qnn_backend = dynamic_cast<QNNBackend *>(this->backend_);
    qnn_backend->afterAllGraphsExecute();
}

const vector<shared_ptr<Tensor>> &QNNGraph::forward(bool autofree) {
    // backend event hook
    this->backend_->onExecuteStart(ops_input_tensors_[op_names_[0]], ops_output_tensors_[op_names_[op_names_.size() - 1]]);


    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
#ifdef SAVECHECK
            for (auto &t : ops_input_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif
#ifdef DEBUGPRINT
            uint64_t t_start = mllm_time_us();
#endif
            ops_[op_name]->execute(ops_input_tensors_[op_name],
                                   ops_output_tensors_[op_name]);

#ifdef SAVECHECK
            for (auto &t : ops_output_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif

#ifdef DEBUGPRINT
            uint64_t t_end = mllm_time_us();
            std::cout << "" << op_name
                      << "       exe_time:" << (t_end - t_start) / 1000.0F << " ms"
                      << std::endl;
#endif
            if (autofree) {
                ops_[op_name]->free(ops_input_tensors_[op_name],
                                    ops_output_tensors_[op_name]);
            }
        } else {
            //            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }

    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

} // namespace mllm
