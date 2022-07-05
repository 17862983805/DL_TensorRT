#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int CUDA_NUM_THREADS = 512;

#define CHECK(status)\
    do\
    {\
        int ret = (status);\
        if(ret != 0)\
        {\
            std::cout << "Cuda failure: "<<ret<<std::endl;\
            abort();\
        }\
    }while (0)

__global__ void get_mt_output_kernel(
        const int numThreads, const float* cls_data, const float* reg_data, const float* obj_data, 
        const int height, const int width,
        const int cls_channels, const int reg_channels, const int obj_channels,
        float* cls_buffer, float* reg_buffer, float* obj_buffer
        )
{   
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < numThreads; index += blockDim.x * gridDim.x){
        int imgid = index / (width * height);
        int ptid = index % (width * width);
        int startloc_cls = imgid * height * width * cls_channels + ptid;
        int startloc_reg = imgid * height * width * reg_channels + ptid;
        int startloc_obj = imgid * height * width * obj_channels + ptid;
        for(int i=0; i< cls_channels; ++i){
            float clsval = cls_data[startloc_cls + i * height * width];
            cls_buffer[startloc_cls + i * height *width] = 1.0 / (1.0 + expf(-clsval));//sigmoid
        }
        for(int i=0; i< obj_channels; ++i){
            float objval = obj_data[startloc_obj + i * height * width];
            obj_buffer[startloc_obj + i * height *width] = 1.0 / (1.0 + expf(-objval));//sigmoid
        }
        
    }
}

extern "C" void get_mt_output(const float* cls_data, const float* reg_data, const float* obj_data,
            const int num_stage_index, const int batchsize, 
            const int cls_channels ,const int reg_channels, const int obj_channels,
            const int featmap_size[][2], const int num_stage,
            float* cls_buffer, float* reg_buffer, float* obj_buffer
        )
{
    int height = featmap_size[num_stage_index][0];
    int width = featmap_size[num_stage_index][1];
    printf("height: %d, width: %d\n", height, width);
    int numThreads = batchsize * height * width;
    int numBlocks = int(((numThreads + CUDA_NUM_THREADS - 1)) / CUDA_NUM_THREADS);
    get_mt_output_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
        numThreads, cls_data, reg_data, obj_data, 
        height, width,
        cls_channels, reg_channels, obj_channels,
        cls_buffer, reg_buffer, obj_buffer);
    
}