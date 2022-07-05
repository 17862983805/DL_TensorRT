#include "NvInfer.h"
#include "common.hpp"
#include <string>

int main()
{
    std::string onnx_file = "../../yolox_tiny_8x8_300e_coco_sim_modify.onnx";
    //std::string engine_file = "../yolox_s.trt";
    std::string engine_file = "../../yolox_tiny_8x8_300e_coco_sim_modify.engine";
    nvinfer1::ICudaEngine *engine;
    
    onnxToTRTModel(onnx_file, engine_file, engine, 1);
    assert(engine != nullptr);
    
    return 1;
}
