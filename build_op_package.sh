export QNN_SDK_ROOT=src/backends/qnn/qualcomm_ai_engine_direct_220/
export ANDROID_NDK_ROOT=/home/huzq85/software/softwareDev/AndroidSDK/ndk/26.3.11579264
export PATH=$PATH:$ANDROID_NDK_ROOT

source src/backends/qnn/HexagonSDK/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh

cd src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
make htp_aarch64 && make htp_v73
