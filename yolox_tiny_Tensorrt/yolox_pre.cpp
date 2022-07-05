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

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25

using namespace nvinfer1;
using namespace std;

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 416;
static const int INPUT_H = 416;
static const int NUM_CLASSES = 80;
static Logger gLogger;

const char* bindingName[10] = {"input",\
                      "801","802","803",\
                      "820","821","822",\
                      "839","840","841"};
std::vector<int>bindingIndex(10);

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows + 0.5;
    cout<<"unpad_w:"<<unpad_w<<","<<"unpad_h:"<<unpad_h<<endl;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 0, 0));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
static void generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.push_back((GridAndStride){g0 * stride, g1 * stride, stride});
            }
        }
        // cout<<"grid_strides:"<<grid_strides.size()<<endl;
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
//    cout<<inter<<endl;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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


static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, 
                                     float prob_threshold, std::vector<Object>& objects)
{
    const int num_anchors = grid_strides.size();//3549

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);//(3549*85)

        // cout<<grid0<<","<<grid1<<","<<stride<<","<<basic_pos<<endl;

        float x_center = feat_blob[basic_pos+0] * stride + grid0;
        float y_center = feat_blob[basic_pos+1] * stride + grid1;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];

        float objectness_max = 0;
        int objectness_max_index;
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++){
            if(feat_blob[basic_pos + 5 + class_idx] > objectness_max){
                objectness_max = feat_blob[basic_pos + 5 + class_idx];
                objectness_max_index = class_idx;
            }
        }
        
        float box_prob = box_objectness * objectness_max;
        // cout<<objectness_max_index<<":"<<objectness_max<<","<<box_prob<<endl;
        if(box_prob > prob_threshold){
            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = objectness_max_index;
            obj.prob = box_prob;
            objects.push_back(obj);
        }
    }
}


float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    std::vector<float> mean_rgb = {123.67500305, 116.27999878, 103.52999878};
    std::vector<float> std_rgb = {58.395, 57.12, 57.375};
    cout<<channels<<","<<img_h<<","<<img_w<<","<<img.total()*3<<endl;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] = 
                    (((float)img.at<cv::Vec3b>(h, w)[c]) - mean_rgb[c]) / std_rgb[c];
            }
        }
    }
    return blob;
}


static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(INPUT_W, strides, grid_strides);
        cout<<"grid_strides:"<<grid_strides.size()<<endl;
        generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}


const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        // cout<<text<<endl;

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("det_res.jpg", image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python3 yolox/deploy/trt.py -n yolox-{tiny, s, m, l, x}' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./yolox ../model_trt.engine -i ../../../assets/dog.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;

    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    cv::imwrite("pr_img.jpg",pr_img);
    std::cout << "blob image" << std::endl;
    float* blob;
    blob = blobFromImage(pr_img);

    string Tensorrt_preprocess_txt_name = "Tensorrt_preprocess.txt";
    if (access(Tensorrt_preprocess_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_preprocess_txt_name.c_str()) == 0){
            cout<<Tensorrt_preprocess_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    ofstream preprocess_out;
    preprocess_out.open(Tensorrt_preprocess_txt_name);
    for (int i = 0; i < pr_img.total()*3; i++){
        preprocess_out<<blob[i]<<"\n";
    }
    preprocess_out.close();

    string onnx_preprocess_txt = "../../checkpoints/yolox_pytorch2onnx_test/onnx_preprocess.txt";
    vector<float>onnx_preprocess;
    ifstream onnx_preprocess_data(onnx_preprocess_txt);
    float d;
    while (onnx_preprocess_data >> d)
    {
        onnx_preprocess.push_back(d);
    }
    string pytorch_inference_txt = "../../checkpoints/yolox_pytorch2onnx_test/pytorch_inference_preprocess.txt";
    vector<float>pytorch_inference_preprocess;
    ifstream pytorch_inference_data(pytorch_inference_txt);
    float d_;
    while (pytorch_inference_data >> d_)
    {
        pytorch_inference_preprocess.push_back(d_);
    }
    cout<<pytorch_inference_preprocess.size()<<endl;

    float max_diff = 0.0;
    for (int i = 0; i < pr_img.total()*3; i++)
    {
        float diff = abs(blob[i] - onnx_preprocess[i]);
        // cout<<blob[i]<<","<<onnx_preprocess[i]<<endl;
        if(diff > max_diff) {
            max_diff = diff;
        }
        // blob[i] = onnx_preprocess[i];
        // blob[i] = pytorch_inference_preprocess[i];
    }
    cout<<"max diff of preprocess:"<<max_diff<<endl;

    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));

    int nbBindings = engine->getNbBindings();
    std::vector<int64_t> mBindBufferSizes(nbBindings);
    void* cudaOutputBuffer[nbBindings];

    cudaStream_t stream;
    const int maxBatchSize = 1;
    vector<vector<int>>output_strides;
    for (int i = 0; i < nbBindings; i++)
    {
        vector<int>stride;
        int64_t totalSize = 1;
        std::cout<<"+++++++++++++++++++"<<std::endl;
        const char* layername = engine->getBindingName(i);
        std::cout<< "getBindingName:" << ":" << layername << std::endl;
        nvinfer1::Dims output_dims = engine->getBindingDimensions(i);
        std::cout<<"The shape of engine model is ("
        <<output_dims.d[0]
        <<","<<output_dims.d[1]
        <<","<<output_dims.d[2]
        <<","<<output_dims.d[3]
        <<")"<<std::endl;

        for (int j = 0; j < output_dims.nbDims; j++) stride.push_back(output_dims.d[i]);
        output_strides.push_back(stride);
        
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        for(int j = 0;j < output_dims.nbDims;++j){
            totalSize *= output_dims.d[j];
        }
        std::cout<<"totalSize:"<<totalSize<<std::endl;
        bindingIndex[i] = engine->getBindingIndex(bindingName[i]);
        cout<<"bindingIndex:"<<bindingIndex[i]<<endl;
        std::cout<<"------------------"<<std::endl;
        mBindBufferSizes[i] = totalSize;
        CHECK(cudaMalloc(&cudaOutputBuffer[i],totalSize * sizeof(float)));
    }
    CHECK(cudaStreamCreate(&stream));
    const int batchSize = 1;
    int inputIndex = 0;
    float *output[nbBindings];
    for(int i=0; i < nbBindings; ++i){
        output[i] = new float[mBindBufferSizes[i]];
    }
    int64_t totalsize = 0;
    for (int i = 1; i < nbBindings; i++)
    {
        totalsize += mBindBufferSizes[i];
    }
    std::cout<<"totalsize: "<<totalsize<<std::endl;
    CHECK(cudaMemcpyAsync(cudaOutputBuffer[inputIndex],blob,
                          mBindBufferSizes[inputIndex] * sizeof(float),
                          cudaMemcpyHostToDevice,stream));
    std::cout<<"start inferencing...."<<std::endl;
    context->enqueue(1,cudaOutputBuffer,stream,nullptr);
    std::cout<<"end inferencing...."<<std::endl;

    std::cout<<"start Move data to CPU...."<<std::endl;
    for(int i = 0;i < nbBindings;++i){
        CHECK(cudaMemcpyAsync(output[i], cudaOutputBuffer[i], mBindBufferSizes[i] * 
                            sizeof(float), cudaMemcpyDeviceToHost,stream));
    }
    std::cout<<"end Move data to CPU...."<<std::endl;

    std::cout<<"start check the Tensorrt inference results...."<<std::endl;
    std::string pytorch_bbox_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/flatten_bbox_preds.txt";
    std::string pytorch_cls_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/flatten_cls_scores.txt";
    std::string pytorch_obj_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/flatten_objectness.txt";
    vector<float>pytorch_bbox_output;
    vector<float>pytorch_cls_output;
    vector<float>pytorch_obj_output;
    ifstream pytorch_bbox_output_data(pytorch_bbox_output_txt);
    ifstream pytorch_cls_output_data(pytorch_cls_output_txt);
    ifstream pytorch_obj_output_data(pytorch_obj_output_txt);
    float bbox_data, cls_data,obj_data;
    while(pytorch_bbox_output_data >> bbox_data){
        pytorch_bbox_output.push_back(bbox_data);
    }
    while(pytorch_cls_output_data >> cls_data){
        pytorch_cls_output.push_back(cls_data);
    }
    while(pytorch_obj_output_data >> obj_data){
        pytorch_obj_output.push_back(obj_data);
    }
    float* pytorch_inference_result = new float[totalsize];
    int pytorch_index = 0;
    for(int i = 0;i < 3549;++i){
        for(int j=i*4;j<i*4+4;++j){
            pytorch_inference_result[pytorch_index] = pytorch_bbox_output[j];
            pytorch_index++;
        }
        for(int j=i*1;j<i*1+1;++j){
            pytorch_inference_result[pytorch_index] = pytorch_obj_output[j];
            pytorch_index++;
        }
        for(int j=i*80;j<i*80+80;++j){
            pytorch_inference_result[pytorch_index] = pytorch_cls_output[j];
            pytorch_index++;
        }
    }

    string onnx_result_txt = "../../checkpoints/yolox_pytorch2onnx_test/onnx_result.txt";
    string Tensorrt_output_txt_name = "Tensorrt_output.txt";
    if (access(Tensorrt_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_output_txt_name.c_str()) == 0){
            cout<<Tensorrt_output_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_output_txt_name_ = "Tensorrt_output_.txt";
    if (access(Tensorrt_output_txt_name_.c_str(),0) == 0){
        if(remove(Tensorrt_output_txt_name_.c_str()) == 0){
            cout<<Tensorrt_output_txt_name_<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_bbox_output_txt_name = "Tensorrt_bbox_output.txt";
    if (access(Tensorrt_bbox_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_bbox_output_txt_name.c_str()) == 0){
            cout<<Tensorrt_bbox_output_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_cls_output_txt_name = "Tensorrt_cls_output.txt";
    if (access(Tensorrt_cls_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_cls_output_txt_name.c_str()) == 0){
            cout<<Tensorrt_cls_output_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_obj_output_txt_name = "Tensorrt_obj_output.txt";
    if (access(Tensorrt_obj_output_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_obj_output_txt_name.c_str()) == 0){
            cout<<Tensorrt_obj_output_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_bbox_output_sigmoid_txt_name = "Tensorrt_bbox_output_sigmoid.txt";
    if (access(Tensorrt_bbox_output_sigmoid_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_bbox_output_sigmoid_txt_name.c_str()) == 0){
            cout<<Tensorrt_bbox_output_sigmoid_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_cls_output_sigmoid_txt_name = "Tensorrt_cls_output_sigmoid.txt";
    if (access(Tensorrt_cls_output_sigmoid_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_cls_output_sigmoid_txt_name.c_str()) == 0){
            cout<<Tensorrt_cls_output_sigmoid_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    string Tensorrt_obj_output_sigmoid_txt_name = "Tensorrt_obj_output_sigmoid.txt";
    if (access(Tensorrt_obj_output_sigmoid_txt_name.c_str(),0) == 0){
        if(remove(Tensorrt_obj_output_sigmoid_txt_name.c_str()) == 0){
            cout<<Tensorrt_obj_output_sigmoid_txt_name<<" has been deleted successfuly!"<<endl;
        }
    }
    ofstream Tensorrt_out;
    ofstream Tensorrt_out_;
    ofstream Tensorrt_bbox_output;
    ofstream Tensorrt_obj_output;
    ofstream Tensorrt_cls_output;
    ofstream Tensorrt_bbox_sigmoid_output;
    ofstream Tensorrt_obj_sigmoid_output;
    ofstream Tensorrt_cls_sigmoid_output;
    vector<float>Tensorrt_bbox_output_data;
    vector<float>Tensorrt_cls_output_data;
    vector<float>Tensorrt_obj_output_data;
    vector<float>Tensorrt_bbox_sigmoid_output_data;
    vector<float>Tensorrt_cls_sigmoid_output_data;
    vector<float>Tensorrt_obj_sigmoid_output_data;
    Tensorrt_out.open(Tensorrt_output_txt_name);
    Tensorrt_out_.open(Tensorrt_output_txt_name_);
    Tensorrt_bbox_output.open(Tensorrt_bbox_output_txt_name);
    Tensorrt_obj_output.open(Tensorrt_obj_output_txt_name);
    Tensorrt_cls_output.open(Tensorrt_cls_output_txt_name);
    Tensorrt_bbox_sigmoid_output.open(Tensorrt_bbox_output_sigmoid_txt_name);
    Tensorrt_obj_sigmoid_output.open(Tensorrt_obj_output_sigmoid_txt_name);
    Tensorrt_cls_sigmoid_output.open(Tensorrt_cls_output_sigmoid_txt_name);
    for (int i = 1; i < nbBindings; i += 3){
        for (int j = 0; j < mBindBufferSizes[i]; j++){
            Tensorrt_cls_output<<output[i][j]<<"\n";
            Tensorrt_out<<output[i][j]<<"\n";
            Tensorrt_cls_output_data.push_back(output[i][j]);
            output[i][j] = 1.0 / (1.0 + exp(-output[i][j]));
            Tensorrt_out_<<output[i][j]<<"\n";
            Tensorrt_cls_sigmoid_output<<output[i][j]<<"\n";
            Tensorrt_cls_sigmoid_output_data.push_back(output[i][j]);
        }
        for (int j = 0; j < mBindBufferSizes[i + 1]; j++){
            Tensorrt_bbox_output<<output[i + 1][j]<<"\n";
            Tensorrt_out<<output[i + 1][j]<<"\n";
            Tensorrt_bbox_output_data.push_back(output[i + 1][j]);
            Tensorrt_out_<<output[i + 1][j]<<"\n";
            Tensorrt_bbox_sigmoid_output<<output[i + 1][j]<<"\n";
            Tensorrt_bbox_sigmoid_output_data.push_back(output[i + 1][j]);
        }
        for (int j = 0; j < mBindBufferSizes[i + 2]; j++){
            Tensorrt_obj_output<<output[i + 2][j]<<"\n";
            Tensorrt_out<<output[i + 2][j]<<"\n";
            Tensorrt_obj_output_data.push_back(output[i + 2][j]);
            output[i +2][j] = 1.0 / (1.0 + exp(-output[i + 2][j]));
            Tensorrt_out_<<output[i + 2][j]<<"\n";
            Tensorrt_obj_sigmoid_output<<output[i + 2][j]<<"\n";
            Tensorrt_obj_sigmoid_output_data.push_back(output[i + 2][j]);
        }
    }
    Tensorrt_out_.close();
    Tensorrt_out.close();
    Tensorrt_bbox_output.close();
    Tensorrt_obj_output.close();
    Tensorrt_cls_output.close();
    Tensorrt_bbox_sigmoid_output.close();
    Tensorrt_obj_sigmoid_output.close();
    Tensorrt_cls_sigmoid_output.close();
    std::string inference_bbox_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/pytorch_inference_bbox_result.txt";
    std::string inference_cls_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/pytorch_inference_cls_result.txt";
    std::string inference_obj_output_txt = "../../checkpoints/yolox_pytorch2onnx_test/pytorch_inference_obj_result.txt";
    vector<float>inference_bbox_output;
    vector<float>inference_cls_output;
    vector<float>inference_obj_output;
    ifstream inference_bbox_output_data(inference_bbox_output_txt);
    ifstream inference_cls_output_data(inference_cls_output_txt);
    ifstream inference_obj_output_data(inference_obj_output_txt);
    float inference_bbox_data, inference_cls_data, inference_obj_data;
    while(inference_bbox_output_data >> inference_bbox_data){
        inference_bbox_output.push_back(inference_bbox_data);
    }
    while(inference_cls_output_data >> inference_cls_data){
        inference_cls_output.push_back(inference_cls_data);
    }
    while(inference_obj_output_data >> inference_obj_data){
        inference_obj_output.push_back(inference_obj_data);
    }
    cout<<"cls_size:"<<Tensorrt_cls_output_data.size()<<","<<inference_cls_output.size()<<endl;
    cout<<"bbox_size:"<<Tensorrt_bbox_output_data.size()<<","<<inference_bbox_output.size()<<endl;
    cout<<"obj_size:"<<Tensorrt_obj_output_data.size()<<","<<inference_obj_output.size()<<endl;
    vector<float>onnx_result;
    ifstream onnx_result_data(onnx_result_txt);
    float data;
    while (onnx_result_data >> data)
    {
        onnx_result.push_back(data);
    }
    vector<float>Tensorrt_output_result;
    ifstream Tensorrt_output_data(Tensorrt_output_txt_name);
    float trt_data;
    while (Tensorrt_output_data >> trt_data)
    {
        Tensorrt_output_result.push_back(trt_data);
    }
    max_diff = 0;
    float diff_all = 0;
    for (int i = 0; i < totalsize; i++)
    {
        float diff = abs(Tensorrt_output_result[i] - onnx_result[i]);
        diff_all += diff;
        if(diff > max_diff) {
            max_diff = diff;
        }
    }
    cout<<"Tensorrt_output_result max diff:"<<max_diff<<endl;
    cout<<"average diff:"<<diff_all / float(totalsize)<<endl;
    float cls_max_diff = 0;
    for (int i = 0; i < Tensorrt_cls_output_data.size(); i++)
    {
        if(abs(Tensorrt_cls_output_data[i] - inference_cls_output[i]) > cls_max_diff)
            cls_max_diff = abs(Tensorrt_cls_output_data[i] - inference_cls_output[i]);
    }
    float bbox_max_diff = 0;
    for (int i = 0; i < Tensorrt_bbox_output_data.size(); i++)
    {
        if(abs(Tensorrt_bbox_output_data[i] - inference_bbox_output[i]) > bbox_max_diff)
            bbox_max_diff = abs(Tensorrt_bbox_output_data[i] - inference_bbox_output[i]);
    }
    float obj_max_diff = 0;
    for (int i = 0; i < Tensorrt_obj_output_data.size(); i++)
    {
        if(abs(Tensorrt_obj_output_data[i] - inference_obj_output[i]) > obj_max_diff)
            obj_max_diff = abs(Tensorrt_obj_output_data[i] - inference_obj_output[i]);
    }
    cout<<"cls_max_diff:"<<cls_max_diff<<endl;
    cout<<"bbox_max_diff:"<<bbox_max_diff<<endl;
    cout<<"obj_max_diff:"<<obj_max_diff<<endl;
    std::cout<<"end check the Tensorrt inference results...."<<std::endl;

    std::cout<<"start save Tensorrt inference results...."<<std::endl;
    std::vector<int> strides = {8, 16, 32};
    float* Tensorrt_result = new float[totalsize];
    vector<vector<float>>Tensorrt_result_stage(3);
    int index_ = 0;
    for (int i = 0; i < 3; i++)
    {
        if(i != 0){
            int stride_pre = INPUT_W / strides[i - 1];
            index_ += stride_pre * stride_pre * 80;
        }
        int stride = INPUT_W / strides[i];
        for (int j = 0; j < stride * stride; j++){
            for (int k = 0; k < 80; k++){
                int index = j + stride * stride * k;
                if(i != 0) index += index_;
                Tensorrt_result_stage[0].push_back(Tensorrt_cls_sigmoid_output_data[index]);
            }
        }
    }
    index_ = 0;
    for (int i = 0; i < 3; i++)
    {
        if(i != 0){
            int stride_pre = INPUT_W / strides[i - 1];
            index_ += stride_pre * stride_pre * 4;
        }
        int stride = INPUT_W / strides[i];
        for (int j = 0; j < stride * stride; j++){
            for (int k = 0; k < 4; k++){
                int index = j + stride * stride * k;
                if(i != 0) index += index_;
                Tensorrt_result_stage[1].push_back(Tensorrt_bbox_sigmoid_output_data[index]);
            }
        }
    }
    index_ = 0;
    for (int i = 0; i < 3; i++)
    {
        if(i != 0){
            int stride_pre = INPUT_W / strides[i - 1];
            index_ += stride_pre * stride_pre * 1;
        }
        int stride = INPUT_W / strides[i];
        for (int j = 0; j < stride * stride; j++){
            for (int k = 0; k < 1; k++){
                int index = j + stride * stride * k;
                if(i != 0) index += index_;
                Tensorrt_result_stage[2].push_back(Tensorrt_obj_sigmoid_output_data[index]);
            }
        }
    }
    int tensorrt_index = 0;
    for(int i = 0;i < 3549;++i){
        for(int j=i*4;j<i*4+4;++j){
            Tensorrt_result[tensorrt_index] = Tensorrt_result_stage[1][j];
            tensorrt_index++;
        }
        for(int j=i*1;j<i*1+1;++j){
            Tensorrt_result[tensorrt_index] = Tensorrt_result_stage[2][j];
            tensorrt_index++;
        }
        for(int j=i*80;j<i*80+80;++j){
            Tensorrt_result[tensorrt_index] = Tensorrt_result_stage[0][j];
            tensorrt_index++;
        }
    }

    std::cout<<"end save Tensorrt inference results...."<<std::endl;

    float max_inference_result_diff = 0;
    for (int i = 0; i < totalsize; i++)
    {
        // cout<<i<<":"<<Tensorrt_result[i]<<","<<pytorch_inference_result[i]<<endl;
        if(abs(Tensorrt_result[i] - pytorch_inference_result[i]) > max_inference_result_diff){
            // cout<<Tensorrt_result[i]<<","<<pytorch_inference_result[i]<<endl;
            max_inference_result_diff = abs(Tensorrt_result[i] - pytorch_inference_result[i]);
        }
    }
    cout<<"max_inference_result_diff:"<<max_inference_result_diff<<endl;

    std::cout<<"start postprocess...."<<std::endl;
    std::vector<Object> objects;
    decode_outputs(Tensorrt_result, objects, scale, img_w, img_h);
    // decode_outputs(pytorch_inference_result, objects, scale, img_w, img_h);
    draw_objects(img, objects, input_image_path);

    std::cout<<"Free space...."<<std::endl;
    for(int i = 0;i < nbBindings;++i){
        CHECK(cudaFree(cudaOutputBuffer[i]));
    }
    
    // Release stream and buffers
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    std::cout<<std::endl;
    for(int i=0; i < nbBindings; ++i){
        delete [] output[i];
    }

    delete [] Tensorrt_result;
    delete [] pytorch_inference_result;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
