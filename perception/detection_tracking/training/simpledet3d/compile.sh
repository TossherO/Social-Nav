#!/bin/bash
# 编译 CUDA 扩展（生成 .so 文件到源码目录下）
# Compile CUDA extensions in-place

cd "$(dirname "$0")"
python setup.py build_ext --inplace
echo "CUDA extensions compiled successfully!"
