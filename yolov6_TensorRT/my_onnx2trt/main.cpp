#include "NvInfer.h"
#include "common.hpp"
#include <string>

int main()
{
    std::string onnx_file = "/home/uisee/YOLOv6/weights/yolov6s.onnx";
    std::string engine_file = "../../yolov6.engine";
    nvinfer1::ICudaEngine *engine;
    
    onnxToTRTModel(onnx_file, engine_file, engine, 1);
    assert(engine != nullptr);
    
    return 1;
}
