/**
 * @file XpTest.hpp
 * @author chenghua.wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-31
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "gtest/gtest.h"
#include "backends/xnnpack/XnnpackBackend.hpp"

using namespace mllm;
using namespace mllm::xnnpack;

class XpTest : public ::testing::Test {
public:
    XpTest() = default;
    ~XpTest() override = default;

    void SetUp() override;

    void TearDown() override;

private:
    Backend *bk_ = nullptr;
    std::shared_ptr<MemoryManager> mm_ = nullptr;
};
