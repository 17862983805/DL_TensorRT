#!/bin/bash

# 请修改protoc为你要使用的版本protoc
export LD_LIBRARY_PATH=/home/uisee/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/trt8cuda102cudnn8/lib64
protoc=/home/uisee/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/trt8cuda102cudnn8/bin/protoc

cd onnx

echo Create directory "pbout"
rm -rf pbout
mkdir -p pbout

$protoc onnx-ml.proto --cpp_out=pbout
$protoc onnx-operators-ml.proto --cpp_out=pbout

echo Copy pbout/onnx-ml.pb.cc to ../src/onnx/onnx-ml.pb.cpp
cp pbout/onnx-ml.pb.cc           ../src/onnx/onnx-ml.pb.cpp

echo Copy pbout/onnx-operators-ml.pb.cc to ../src/onnx/onnx-operators-ml.pb.cpp
cp pbout/onnx-operators-ml.pb.cc ../src/onnx/onnx-operators-ml.pb.cpp

echo Copy pbout/onnx-ml.pb.h to ../src/onnx/onnx-ml.pb.h
cp pbout/onnx-ml.pb.h           ../src/onnx/onnx-ml.pb.h

echo Copy pbout/onnx-operators-ml.pb.h to ../src/onnx/onnx-operators-ml.pb.h
cp pbout/onnx-operators-ml.pb.h ../src/onnx/onnx-operators-ml.pb.h

echo Remove directory "pbout"
rm -rf pbout