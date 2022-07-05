cmake_minimum_required(VERSION 2.6)

project(yolox)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(lib_DIR /home/uisee/vcpkg/installed/x64-linux/lib)
link_directories(${lib_DIR})

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
#include_directories(/usr/local/cuda/cudnn/v8.0.4/include)
#link_directories(/usr/local/cuda/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/home/uisee/TensorRT-7.2.1.6/include)
link_directories(/home/uisee/TensorRT-7.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA_ADD_EXECUTABLE(yolox ${PROJECT_SOURCE_DIR}/yolox.cpp ${PROJECT_SOURCE_DIR}/plugin_cuda_functions.cu)
CUDA_ADD_EXECUTABLE(yolox ${PROJECT_SOURCE_DIR}/src/yolox_run.cpp  ${PROJECT_SOURCE_DIR}/src/yolox.cpp)
# add_executable(yolox ${PROJECT_SOURCE_DIR}/yolox_pre.cpp)
# add_executable(yolox ${PROJECT_SOURCE_DIR}/yolox_coco.cpp)
target_link_libraries(yolox nvinfer)
target_link_libraries(yolox cudart)
target_link_libraries(yolox libjsoncpp.a)
target_link_libraries(yolox ${OpenCV_LIBS})

add_definitions(-O2 -pthread)


