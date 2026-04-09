#!/bin/bash

echo "=== 开始编译船舶重心计算程序 ==="

# 1. 清理旧编译
rm -rf build_linux
mkdir build_linux
cd build_linux

# 2. 编译
cmake ..
make -j$(nproc)

cd ..

echo "=== 编译完成，开始计算 ==="

# 3. 运行
./build_linux/ShipGravityLab

echo "=== 计算完成 ==="