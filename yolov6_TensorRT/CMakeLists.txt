cmake_minimum_required(VERSION 2.6)

project(yolov6)

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

# tensorrt
include_directories(/home/uisee/TensorRT-7.2.1.6/include)
link_directories(/home/uisee/TensorRT-7.2.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(yolov6 ${PROJECT_SOURCE_DIR}/src/main.cpp  ${PROJECT_SOURCE_DIR}/src/yolov6.cpp)
target_link_libraries(yolov6 nvinfer)
target_link_libraries(yolov6 cudart)
target_link_libraries(yolov6 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)