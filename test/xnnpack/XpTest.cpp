/**
 * @file XpTest.cpp
 * @author chenghua.wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-31
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "XpTest.hpp"
#include "xnnpack/XnnpackBackend.hpp"

void XpTest::SetUp() {
    bk_ = new XnnpackBackend(mm_);
    std::cout << "XnnpackBackend SetUp()" << std::endl;
}

void XpTest::TearDown() {
    delete bk_;
    std::cout << "TearDown TearDown()" << std::endl;
}
