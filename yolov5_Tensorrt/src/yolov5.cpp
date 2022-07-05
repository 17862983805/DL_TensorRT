#include "../include/yolov5.h"
#include <chrono>
#include <iostream>
#include "../include/common.h"
#include <fstream>
#include <unistd.h>

static const int DEVICE  = 0;

YOLOV5::YOLOV5(const int INPUT_H, 
           const int INPUT_W,
           const std::string& _engine_file):
    input_dim2(INPUT_H),
    input_dim3(INPUT_W),
    input_bufsize(input_dim0 * input_dim1 *
                input_dim2 * input_dim3 *
                sizeof(float)),
    output_bufsize(output_dim0 * 
                output_dim1 *
                output_dim2 *
                sizeof(float)),
    engine_file(_engine_file)
{
    init_context();
    std::cout<<"Inference ["<<input_dim2<<"x"<<input_dim3<<"] constructed"<<std::endl;
    init_done = true;
}

YOLOV5::~YOLOV5(){
    if(init_done){
        destroy_context();
        std::cout<<"Context destroyed for ["<<input_dim2<<"x"<<input_dim3<<"]"<<std::endl;
    }
}

void YOLOV5::init_context(){
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    bool cudart_ok = true;
    cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(cudart_ok);

    /* Allocate memory for inference */
    const size_t buf_num = input_layers.size() + output_layers.size();
    std::cout<<"buf_num:"<<buf_num<<std::endl;
    cudaOutputBuffer = std::vector<void *>(buf_num, nullptr);
    hostOutputBuffer = std::vector<void *>(buf_num, nullptr);

    for(auto & layer : input_layers)
    {
        const std::string & name = layer.first;
        int index = create_binding_memory(name);
        input_layers.at(name) = index; /* init binding index */
    }

    for(auto & layer : output_layers)
    {
        const std::string & name = layer.first;
        int index = create_binding_memory(name);
        output_layers.at(name) = index; /* init binding index */
    }
}

void YOLOV5::destroy_context(){
    bool cudart_ok = true;

    /* Release TensorRT */
    if(context)
    {
        context->destroy();
        context = nullptr;
    }

    if(engine)
    {
        engine->destroy();
        engine = nullptr;
    }

    if(stream) cudaStreamDestroy(stream);

    CHECK_CUDA_ERROR(cudart_ok);

    /* Release memory for inference */
    for(int i=0; i < (int)cudaOutputBuffer.size(); i++)
    {
        if(cudaOutputBuffer[i])
        {
            cudaFree(cudaOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            cudaOutputBuffer[i] = nullptr;
        }
    }
    for(int i=0; i < (int)hostOutputBuffer.size(); i++)
    {
        if(hostOutputBuffer[i])
        {
            free(hostOutputBuffer[i]);
            CHECK_CUDA_ERROR(cudart_ok);
            hostOutputBuffer[i] = nullptr;
        }
    }
}

int YOLOV5::create_binding_memory(const std::string& bufname){
    assert(engine != nullptr);

    const int devbuf_num  = static_cast<int>(cudaOutputBuffer.size());
    const int hostbuf_num = static_cast<int>(hostOutputBuffer.size());

    int index = engine->getBindingIndex(bufname.c_str());

    size_t elem_size = 0;
    switch (engine->getBindingDataType(index)){
        case nvinfer1::DataType::kFLOAT:
            elem_size = sizeof(float); break;
        case nvinfer1::DataType::kHALF:
            elem_size = sizeof(float) >> 1; break;
        case nvinfer1::DataType::kINT8:
            elem_size = sizeof(int8_t); break;
        case nvinfer1::DataType::kINT32:
            elem_size = sizeof(int32_t); break;
        default:
            ; /* fallback */
    }
    assert(elem_size != 0);

    size_t elem_count = 0;
    nvinfer1::Dims dims = engine->getBindingDimensions(index);
    for (int i = 0; i < dims.nbDims; i++){
        if (0 == elem_count){
            elem_count = dims.d[i];
        }
        else{
            elem_count *= dims.d[i];
        }
    }

    size_t buf_size = elem_count * elem_size;
    assert(buf_size != 0);

    void * device_mem;
    bool cudart_ok = true;

    cudaMalloc(&device_mem, buf_size);
    CHECK_CUDA_ERROR(cudart_ok);
    assert(cudart_ok);

    cudaOutputBuffer[index] = device_mem;

    void * host_mem = malloc( buf_size );
    assert(host_mem != nullptr);

    hostOutputBuffer[index] = host_mem;

    printf("Created host and device buffer for %s "   \
        "with bindingIndex[%d] and size %lu bytes.\n", \
        bufname.c_str(), index, buf_size );

    return index;
}

void* YOLOV5::get_infer_bufptr(const std::string& bufname, bool is_device){
    assert(init_done);

    int index = -1;

    if (bufname == input)
    {
        index = input_layers.at(bufname);
    }
    else
    {
        index = output_layers.at(bufname);
    }

    return (is_device ? cudaOutputBuffer.at(index) : hostOutputBuffer.at(index));
}

void YOLOV5::do_inference(cv::Mat& image, cv::Mat& dst){
    assert(context != nullptr);

    void* host_input = get_infer_bufptr(input, false);
    void* device_input = get_infer_bufptr(input, true);

    void* host_output = get_infer_bufptr(output, false);
    void* device_output = get_infer_bufptr(output, true);
    
    float* d2i_ = pre_process(image, input_dim2, input_dim3, static_cast<float *>(host_input));

    cudaMemcpyAsync(device_input, host_input, input_bufsize,
                        cudaMemcpyHostToDevice, stream);
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();

    context->enqueue(batchsize, &cudaOutputBuffer[0], stream, nullptr);

    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    cudaMemcpyAsync(host_output, device_output, output_bufsize,
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(res_ok);

    assert(res_ok);
    std::cout<<"Model enqueue done!"<<std::endl;

    auto output_dims = engine->getBindingDimensions(1);
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    post_process(static_cast<float *>(host_input),
                 static_cast<float *>(host_output),
                 d2i_,
                 output_numbox,
                 output_numprob,
                 num_classes,
                 dst);
    std::cout<<"Post-process done!"<<std::endl;
}

float* YOLOV5::pre_process(cv::Mat& image, int input_height, int input_width,float* input_data_host){
    // 通过双线性插值对图像进行resize
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    /*
    C++ 不支持在函数外返回局部变量的地址，除非定义局部变量为 static 变量。
    https://www.runoob.com/cplusplus/cpp-return-arrays-from-function.html
    */
    static float i2d[6], d2i[6];
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
    // cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    // for(int k=0;k<6;++k){
    //     std::cout<<"d2i["<<k<<"]:"<<d2i[k]<<std::endl;
    // }
    return d2i;
}

void YOLOV5::post_process(float* input_data_host,
                          float* output_data_host,
                          float* d2i,
                          int output_numbox,
                          int output_numprob,
                          int num_classes,
                          cv::Mat& image){
    // decode box：从不同尺度下的预测狂还原到原输入图上(包括:预测框，类被概率，置信度）
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    for(int i = 0; i < output_numbox; ++i){
        float* ptr = output_data_host + i * output_numprob;
        float objness = ptr[4];
        if(objness < confidence_threshold)
            continue;

        float* pclass = ptr + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx     = ptr[0];
        float cy     = ptr[1];
        float width  = ptr[2];
        float height = ptr[3];

        // 预测框
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 对应图上的位置
        float image_base_left   = d2i[0] * left   + d2i[2];
        float image_base_right  = d2i[0] * right  + d2i[2];
        float image_base_top    = d2i[0] * top    + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms非极大抑制
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
        float cross_left   = std::max(a[0], b[0]);
        float cross_top    = std::max(a[1], b[1]);
        float cross_right  = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                         + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for(int i = 0; i < box_result.size(); ++i){
        auto& ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = cocolabels[class_label];
        auto caption   = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    // cv::imwrite("yolov5s_result.jpg", image);
}