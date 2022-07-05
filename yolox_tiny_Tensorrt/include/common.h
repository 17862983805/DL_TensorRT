#ifndef COMMON_H
#define COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include "NvInfer.h"
#include <sstream>
#include <iostream>

#define CHECK_CUDA_FATAL(call)                                  \
     do {                                                        \
          cudaError_t result_ = (call);                           \
          if ( result_ != cudaSuccess )                           \
          {                                                       \
             printf(#call "failed (@loc: %d): %s \n",   \
                  __LINE__, cudaGetErrorString(result_));     \
             abort();                                            \
          }                                                       \
     } while (0)

#define CHECK_CUDA_ERROR(res_ok)                                \
    do {                                                        \
         cudaError_t status_ = cudaGetLastError();               \
         if ( status_ != cudaSuccess  )                          \
         {                                                       \
              printf("Cuda failure (@loc: %d): %s\n",    \
              __LINE__, cudaGetErrorString(status_));     \
              res_ok = false;                                     \
         }                                                       \
    } while (0)

// class Logger : public nvinfer1::ILogger
// {
// public:
//     Logger(Severity severity = Severity::kWARNING)
//            : reportableSeverity(severity)
//     {
//     }

//     void log(Severity severity, const char* msg) override
//     {
//         // suppress messages with severity enum value greater than the reportable
//         if (severity > reportableSeverity)
//             return;

//         switch (severity)
//         {
//             case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
//             case Severity::kERROR: std::cerr << "ERROR: "; break;
//             case Severity::kWARNING: std::cerr << "WARNING: "; break;
//             case Severity::kINFO: std::cerr << "INFO: "; break;
//             default: std::cerr << "UNKNOWN: "; break;
//         }
//         std::cerr << msg << std::endl;
//     }
//     Severity reportableSeverity;
// };

static Logger gLogger;


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

#endif//COMMON_H