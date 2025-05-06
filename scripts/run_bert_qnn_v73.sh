#!/bin/bash
adb shell mkdir -p /data/local/tmp/mllm/qnn-lib

adb push ../assets/ /data/local/tmp/mllm/assets/


if ! adb shell [ -f "/data/local/tmp/mllm/models/gte-small-fp32.mllm" ]; then
    adb push ../models/gte-small-fp32.mllm "/data/local/tmp/mllm/models/gte-small-fp32.mllm"
else
    echo "gte-small-fp32.mllm file already exists"
fi


if ! adb shell [ -f "/data/local/tmp/mllm/models/gte_vocab.mllm" ]; then
    adb push ../vocab/gte_vocab.mllm "/data/local/tmp/mllm/vocab/gte_vocab.mllm"
else
    echo "gte_vocab.mllm file already exists"
fi

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

adb push ../bin-arm/demo_bert /data/local/tmp/mllm/bin/
adb shell "cd /data/local/tmp/mllm/bin && export LD_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && export ADSP_LIBRARY_PATH=/data/local/tmp/mllm/qnn-lib && ./demo_bert"