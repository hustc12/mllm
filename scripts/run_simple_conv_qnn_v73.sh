#!/bin/bash

# adb shell mkdir -p /data/local/tmp/mllm/vocab
adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

# adb push ../vocab/qwen_vocab.mllm /data/local/tmp/mllm/vocab/

adb push ../models/simple_conv_model.mllm "/data/local/tmp/mllm/models/simple_conv_model.mllm"

# if ! adb shell [ -f "/data/local/tmp/mllm/models/simple_conv_model.mllm" ]; then
#     adb push ../models/simple_conv_model.mllm "/data/local/tmp/mllm/models/simple_conv_model.mllm"
# else
#     echo "simple_conv_model.mllm file already exists"
# fi



LIBPATH=../src/backends/qnn/qualcomm_ai_engine_direct_220/
ANDR_LIB=$LIBPATH/lib/aarch64-android
OP_PATH=../src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build
DEST=/data/local/tmp/mllm/qnn-lib

adb push $ANDR_LIB/libQnnHtp.so $DEST
adb push $ANDR_LIB/libQnnHtpV73Stub.so $DEST
adb push $ANDR_LIB/libQnnHtpPrepare.so $DEST
adb push $ANDR_LIB/libQnnHtpProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpOptraceProfilingReader.so $DEST
adb push $ANDR_LIB/libQnnHtpV73CalculatorStub.so $DEST
adb push $LIBPATH/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so $DEST
adb push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_CPU.so
adb push $OP_PATH/hexagon-v73/libQnnLLaMAPackage.so $DEST/libQnnLLaMAPackage_HTP.so


if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi

adb push ../bin-arm/demo_simple_conv /data/local/tmp/mllm/bin/
adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./demo_simple_conv"
