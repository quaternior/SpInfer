#!/bin/bash
# 创建下载目录
mkdir -p ${SpInfer_HOME}/third_party/cusparselt

# 下载 cuSparseLt
echo "Downloading cuSparseLt..."
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz -P ${SpInfer_HOME}/third_party/cusparselt

# 解压文件
echo "Extracting cuSparseLt..."
cd ${SpInfer_HOME}/third_party/cusparselt
tar -xf libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz --strip-components=1

# 清理下载的压缩包
rm libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz

