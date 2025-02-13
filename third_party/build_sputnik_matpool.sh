#!/bin/bash

# 设置基础环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH

# 编译安装 glog
cd ${SpInfer_HOME}/third_party/glog
rm -rf build
mkdir -p build
cd build

# 配置和编译 glog，使用静态库避免符号链接问题
cmake .. \
    -DCMAKE_INSTALL_PREFIX=${SpInfer_HOME}/third_party/glog/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DWITH_GFLAGS=OFF \
    -DCMAKE_BUILD_TYPE=Release

make -j12
make install

# 设置 glog 环境变量
export GLOG_PATH=${SpInfer_HOME}/third_party/glog
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$GLOG_PATH/build/include
export LIBRARY_PATH=$LIBRARY_PATH:$GLOG_PATH/build/lib

# 编译 sputnik
cd ${SpInfer_HOME}/third_party/sputnik
rm -rf build
mkdir -p build
cd build

# 配置和编译 sputnik，使用静态 glog
cmake .. \
    -DGLOG_INCLUDE_DIR=$GLOG_PATH/build/include \
    -DGLOG_LIBRARY=$GLOG_PATH/build/lib/libglog.a \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TEST=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DCUDA_ARCHS="89;86;80" \
    -DCMAKE_INSTALL_RPATH="${SpInfer_HOME}/third_party/glog/build/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE


make -j12
