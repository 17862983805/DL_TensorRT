cmake_minimum_required(VERSION 3.5)
project(my_onnx2trt)

set(CMAKE_CXX_STANDARD 14)
set( CMAKE_BUILD_TYPE "Debug" )

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /home/uisee/TensorRT-7.2.1.6)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${COMMON_INCLUDE})
add_executable(onnx2trt main.cpp)

target_link_libraries(onnx2trt ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})


#cmake_minimum_required(VERSION 3.5)
#project(iou_tracker_test)
#
#set(CMAKE_CXX_STANDARD 14)
#set( CMAKE_BUILD_TYPE "Debug" )
#
## OpenCV
#find_package(OpenCV REQUIRED)
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#
## CUDA
#find_package(CUDA REQUIRED)
#message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
#message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
#
## TensorRT
#set(TENSORRT_ROOT /usr/src/tensorrt/)
#find_path(TENSORRT_INCLUDE_DIR NvInfer.h
#        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
#message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
#find_library(TENSORRT_LIBRARY_INFER nvinfer
#        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
#        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES lib lib64 lib/x64)
#set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
#message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")
#
#include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE})
#add_executable(iou_tracker_test main.cpp Iou_tracker.cpp Iou_tracker.h tracker.h
#        ./deep/common.hpp
#        ./deep/logging.h
#        ./deep/reid_net.cpp
#        ./deep/reid_net.h)
#
#target_link_libraries(iou_tracker_test ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} )