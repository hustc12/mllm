export QNN_SDK_ROOT=../src/backends/qnn/sdk/
export ANDROID_NDK_ROOT=/home/huzq85/software/softwareDev/AndroidSDK/ndk/26.3.11579264
export PATH=$PATH:$ANDROID_NDK_ROOT

source /home/huzq85/software/softwareDev/Qualcomm/HexagonSDK/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh

cd ../src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
make htp_aarch64 && make htp_v73
