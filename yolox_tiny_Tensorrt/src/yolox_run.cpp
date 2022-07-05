#include <chrono>
#include <gflags/gflags.h>

/* The Only header file for yolox inference */
#include "../include/yolox.h"

// const char* __yolox_trt_usage_string =
//  "\n  Usage:"
//  "\n    yolox_test         --H          [image height]"
//  "\n                       --W          [image width]"
//  "\n                       --model      [model file path]"
//  "\n                       --img_path   [image path]"
//  "\n                       --vis        0 (default)"
//  "\n                       --dump_res   true (default)"
//  "\n";

// DEFINE_int32(H,        0,     "image height");
// DEFINE_int32(W,        0,     "image width");
// DEFINE_string(model,   "",    "trt engine file");
// DEFINE_string(img_path, "",   "input image path");
// DEFINE_int32(vis,      0,     "visualize results");
// DEFINE_bool(dump_res,  true,  "dump image for analysis");

int main(int argc, char **argv)
{
    /* Parse command line options */
    // gflags::SetUsageMessage(__yolox_trt_usage_string);
    // gflags::ParseCommandLineFlags(&argc, &argv, true /* remove_flags */);
    
    // std::cout << "Model path: " << FLAGS_model << std::endl               
    //           << "Input image path: " << FLAGS_img_path << std::endl         
    //           << "Visualize: " << FLAGS_vis << std::endl                  
    //           << "Dump res for analysis: " << FLAGS_dump_res << std::endl
    //           << std::endl << "\n";
    
    /* Read image */
    const int FLAGS_vis = 0;
    const int FLAGS_dump_res = 1;
    const std::string FLAGS_img_path = "/home/uisee/disk/dollymonitor_log_file_data_root/mmdetection/demo/demo.jpg";
    const std::string FLAGS_model = "/home/uisee/disk/dollymonitor_log_file_data_root/mmdetection/yolox_tiny_Tensorrt/yolox_tiny_8x8_300e_coco_sim.engine";
    cv::Mat image = cv::imread(FLAGS_img_path);
    if(image.empty()){
        std::cout<<"Input image path wrong!!"<<std::endl;
        return -1;
    }
    yolox::Yolo_trt* yolox_instance = new yolox::Yolo_trt(FLAGS_model);
    
    /* End-to-end infer */
    cv::Mat dst = image.clone();
    yolox_instance->do_inference(image, dst);
    
    if(FLAGS_vis)
    {
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }
    
    if(FLAGS_dump_res)
    {
        std::string save_path = "./yolox.jpg";
        cv::imwrite(save_path, dst);
    }
    
    if(yolox_instance)
    {
        delete yolox_instance;
        yolox_instance = nullptr;
    }
    return 0;
}
