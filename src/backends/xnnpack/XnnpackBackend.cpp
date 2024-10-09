#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Backend.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "xnnpack.h"
#include "xnnpack/Ops/XpBinary.hpp"
#include "backends/xnnpack/XpMemoryManager.hpp"
#include "xnnpack/allocator.h"
#include "xnnpack/subgraph.h"

namespace mllm {

class XpBackendCreator : public BackendCreator {
    Backend *create(BackendConfig config) override {
        // create xnnpack
        if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
            ::mllm::xnnpack::Log::error("failed to initialize XNNPACK");
            return nullptr;
        }

        auto mm = std::make_shared<::mllm::xnnpack::XpMemoryManager>();
        return new ::mllm::xnnpack::XnnpackBackend(mm);
    };
};

void registerXNNBackendCreator() {
    ::mllm::xnnpack::Log::info("xnnpack backend registered");
    InsertBackendCreatorMap(MLLM_XNNPACK, std::make_shared<XpBackendCreator>());
}
} // namespace mllm

namespace mllm::xnnpack {

XnnpackModelRuntime::XnnpackModelRuntime(int32_t num_threads) :
    num_threads_(num_threads), model_(nullptr, xnn_delete_subgraph) {
    xnn_delete_runtime(runtime_);
    threadpool_ = pthreadpool_create(num_threads_);
}

XnnpackModelRuntime::~XnnpackModelRuntime() {
    if (runtime_) {
        xnn_delete_runtime(runtime_);
    }
    if (threadpool_) {
        pthreadpool_destroy(threadpool_);
    }

    // not release output external memory
    for (auto i = 0; i < external_values_.size(); ++i) {
        if ((model_->values[i].flags & ((uint32_t)XNN_VALUE_FLAG_EXTERNAL_INPUT)) == 1) {
            xnn_release_simd_memory(uuid_2_externals_v_[i].data);
        }
    }
}

bool XnnpackModelRuntime::createModel(const xnn_subgraph_t &model_factory) {
    model_.reset(model_factory);
    if (!model_) {
        Log::error("failed to create model");
        return false;
    }

    for (uint32_t i = 0; i < model_->num_values; ++i) {
        // if not external values. ignore alloc memory
        if ((model_->values[i].flags & ((uint32_t)XNN_VALUE_FLAG_EXTERNAL_INPUT | (uint32_t)XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
            continue;
        }

        // if already alloced by user, ignore alloc memory
        if (uuid_2_externals_v_[i].data) {
            external_values_.push_back(xnn_external_value{i, uuid_2_externals_v_[i].data});
            continue;
        }

        // Make a buffer for this external value.
        size_t size = xnn_tensor_get_size(&model_->values[i]) + XNN_EXTRA_BYTES;
        external_values_.push_back(
            xnn_external_value{i, xnn_allocate_zero_simd_memory(size)});
    }

    return model_ != nullptr;
}

bool XnnpackModelRuntime::createRuntime(uint32_t flags) {
    assert(!runtime_);
    return xnn_status_success == xnn_create_runtime_v4(model_.get(), nullptr, nullptr, threadpool_, flags, &runtime_);
}

bool XnnpackModelRuntime::reshapeRuntime() {
    return xnn_status_success == xnn_reshape_runtime(runtime_);
}

bool XnnpackModelRuntime::setupRuntime() {
    return xnn_status_success == xnn_setup_runtime_v2(runtime_, external_values_.size(), external_values_.data());
}

bool XnnpackModelRuntime::invoke() {
    return xnn_status_success == xnn_invoke_runtime(runtime_);
}

void XnnpackModelRuntime::resetUuidExternalValuesMap(const std::unordered_map<uint32_t, xnn_external_value> &ext_vals) {
    uuid_2_externals_v_ = ext_vals;
}

XnnpackBackend::XnnpackBackend(std::shared_ptr<MemoryManager> mm, const XnnpackBackendOpts &opts) :
    Backend(mm), opts_(opts) {
    model_runtime_ = std::make_shared<XnnpackModelRuntime>(opts_.num_threads);
}

XnnpackBackend::~XnnpackBackend() {
    // TODO
}

bool XnnpackBackend::addCreator(OpType t, Creator *c) {
    if (map_op_creator_.count(t)) {
        Log::error("{} op has been added to this backend.", (int32_t)t);
        return false;
    }
    map_op_creator_.insert({t, c});
    return true;
}

Op *XnnpackBackend::opCreate(const OpParam &op_param, string name, int thread_count) {
    auto op_type = OpType(op_param.find("type")->second);
    auto iter = map_op_creator_.find(op_type);

    if (thread_count) {
        Log::warn("Xnnpack use global thread pool. thread_count is set to {}, but not used.", thread_count);
    }

    if (iter == map_op_creator_.end()) {
        Log::error("Op is not supported yet.");
        return nullptr;
    }
    auto op = iter->second->create(op_param, this, name, thread_count);
    return op;
}

TensorFunction *XnnpackBackend::funcCreate(TensorFuncType type) {
    // TODO
    return nullptr;
}

void XnnpackBackend::registerOps() {
    addCreator(ADD, (XnnpackBackend::Creator *)(new XpAddCreator()));
}

void XnnpackBackend::registerFuncs() {
    // TODO
}

std::shared_ptr<XnnpackModelRuntime> XnnpackBackend::getModelRuntime() {
    return model_runtime_;
}

std::shared_ptr<XnnpackModelRuntime> XnnpackBackend::recreateModelRuntime(int thread_count) {
    model_runtime_ = std::make_shared<XnnpackModelRuntime>(thread_count);

    // set external values
    model_runtime_->resetUuidExternalValuesMap(uuid_2_externals_v_);

    return model_runtime_;
}

xnn_subgraph_t XnnpackBackend::getXnnSubgraph() {
    return subgraph_;
}

void XnnpackBackend::registerExternalValue(uint32_t uuid, const xnn_external_value &ext_v) {
    if (uuid_2_externals_v_.count(uuid)) {
        Log::error("when reigster a external value, found exists uuid: {}", uuid);
        exit(-1);
    }

    uuid_2_externals_v_.insert({uuid, ext_v});
}

xnn_datatype XnnpackBackend::mllmDType2XnnDType(DataType mllm_dtype) {
    switch (mllm_dtype) {
    case MLLM_TYPE_F32:
        return xnn_datatype_fp32;
    case MLLM_TYPE_F16:
        return xnn_datatype_fp16;
    case MLLM_TYPE_I32:
        return xnn_datatype_int32;
    default:
        return xnn_datatype_invalid;
    }
    return xnn_datatype_invalid;
}
} // namespace mllm::xnnpack