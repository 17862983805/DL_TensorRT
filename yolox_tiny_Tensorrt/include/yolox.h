#ifndef YOLOX_TRT_H
#define YOLOX_TRT_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <unistd.h>
#include <cmath>

namespace yolox
{
    class Yolo_trt
    {
    public:
        // using mode_data_t = float;

        // std::string input_blob_name = "input";
        // std::vector<std::string> cls_output_blob_name = {"801", "820", "839"};
        // std::vector<std::string> reg_output_blob_name = {"802", "821", "840"};
        // std::vector<std::string> obj_output_blob_name = {"803", "822", "841"};

        const int batchsize = 1;
        const char* input = "input";
        const char* det_cls0 = "801";
        const char* det_cls1 = "820";
        const char* det_cls2 = "839";
        const char* det_bbox0 = "802";
        const char* det_bbox1 = "821";
        const char* det_bbox2 = "840";
        const char* det_obj0 = "803";
        const char* det_obj1 = "822";
        const char* det_obj2 = "841";
        
        // const char* det_cls0 = "det_cls0";
        // const char* det_cls1 = "det_cls1";
        // const char* det_cls2 = "det_cls2";
        // const char* det_bbox0 = "802";
        // const char* det_bbox1 = "821";
        // const char* det_bbox2 = "840";
        // const char* det_obj0 = "det_obj0";
        // const char* det_obj1 = "det_obj1";
        // const char* det_obj2 = "det_obj2";

        Yolo_trt(const std::string& _engine_file);
        ~Yolo_trt();

        virtual void do_inference(cv::Mat& image, cv::Mat& dst);

    private:
        /* Model inputs and outputs constants */
        const int input_dim0 = 1;              /* N */
        const int input_dim1 = 3;             /* C */
        const int input_dim2 = -1;             /* H */
        const int input_dim3 = -1;             /* W */
        const size_t input_bufsize = -1;
        
        const int det_cls0_dim0 = 1;
        const int det_cls0_dim1 = 80;
        const int det_cls0_dim2 = -1;
        const int det_cls0_dim3 = -1;
        const size_t det_cls0_bufsize = -1;
        
        const int det_cls1_dim0 = 1;
        const int det_cls1_dim1 = 80;
        const int det_cls1_dim2 = -1;
        const int det_cls1_dim3 = -1;
        const size_t det_cls1_bufsize = -1;
        
        const int det_cls2_dim0 = 1;
        const int det_cls2_dim1 = 80;
        const int det_cls2_dim2 = -1;
        const int det_cls2_dim3 = -1;
        const size_t det_cls2_bufsize = -1;

        const int det_bbox0_dim0 = 1;
        const int det_bbox0_dim1 = 4;
        const int det_bbox0_dim2 = -1;
        const int det_bbox0_dim3 = -1;
        const size_t det_bbox0_bufsize = -1;
        
        const int det_bbox1_dim0 = 1;
        const int det_bbox1_dim1 = 4;
        const int det_bbox1_dim2 = -1;
        const int det_bbox1_dim3 = -1;
        const size_t det_bbox1_bufsize = -1;
        
        const int det_bbox2_dim0 = 1;
        const int det_bbox2_dim1 = 4;
        const int det_bbox2_dim2 = -1;
        const int det_bbox2_dim3 = -1;
        const size_t det_bbox2_bufsize = -1;

        const int det_obj0_dim0 = 1;
        const int det_obj0_dim1 = 1;
        const int det_obj0_dim2 = -1;
        const int det_obj0_dim3 = -1;
        const size_t det_obj0_bufsize = -1;
        
        const int det_obj1_dim0 = 1;
        const int det_obj1_dim1 = 1;
        const int det_obj1_dim2 = -1;
        const int det_obj1_dim3 = -1;
        const size_t det_obj1_bufsize = -1;
        
        const int det_obj2_dim0 = 1;
        const int det_obj2_dim1 = 1;
        const int det_obj2_dim2 = -1;
        const int det_obj2_dim3 = -1;
        const size_t det_obj2_bufsize = -1;

        const std::string engine_file;
        cudaStream_t stream = nullptr;
        nvinfer1::ICudaEngine *engine = nullptr;
        nvinfer1::IExecutionContext *context = nullptr;
        std::vector<void*> cudaOutputBuffer;
        std::vector<void*> hostOutputBuffer;

        std::map<std::string, int> input_layers {
            /* name,         bufferIndex */
            { input,              -1      }
        };
        std::map<std::string, int> output_layers {
            /* name,         bufferIndex */
            { det_cls0,        -1      },
            { det_cls1,        -1      },
            { det_cls2,        -1      },
            { det_bbox0,       -1      },
            { det_bbox1,       -1      },
            { det_bbox2,       -1      },
            { det_obj0,        -1      },
            { det_obj1,        -1      },
            { det_obj2,        -1      },
        };

        std::vector<std::vector<int>> det_output_stride;

        // std::vector<int>det_output_buffersize(10);

        bool init_done  =false;

        void init_context();
    
        void destroy_context();

        void pre_process(cv::Mat& img, float* host_input_blob);

        void* get_infer_bufptr(const std::string& bufname, bool is_device);

        int create_binding_memory(const std::string& bufname);

        void post_process(float* host_det_cls0, float* host_det_cls1, float* host_det_cls2,
                      float* host_det_bbox0, float* host_det_bbox1, float* host_det_bbox2, 
                      float* host_det_obj0, float* host_det_obj1, float* host_det_obj2,
                      float r, cv::Mat& out_img);
    };
}

#endif//YOLOX_TRT_H