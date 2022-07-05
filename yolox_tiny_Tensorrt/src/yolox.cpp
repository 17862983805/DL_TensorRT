#include "../include/yolox.h"
#include <chrono>
#include <iostream>
#include "../include/common.h"
#include <fstream> 

static const float mean_rgb[3] = {123.67500305, 116.27999878, 103.52999878};
static const float std_rgb[3]  = {58.395, 57.12, 57.375};
static const int stride[3] = {8, 16, 32};

static const float confThreshold = 0.25;
static const float nmsThreshold = 0.45;

static const int DEVICE  = 0;

static const int INPUT_H = 416;
static const int INPUT_W = 416;

namespace yolox
{
    Yolo_trt::Yolo_trt(const std::string& _engine_file):
        input_dim2(INPUT_H),
        input_dim3(INPUT_W),
        input_bufsize(input_dim0 * input_dim1 * 
                      input_dim2 * input_dim3 * 
                      sizeof(float)),
        det_cls0_dim2(INPUT_H/stride[0]),
        det_cls0_dim3(INPUT_W/stride[0]),
        det_cls0_bufsize(det_cls0_dim0 * det_cls0_dim1 *
                         det_cls0_dim2 * det_cls0_dim3 *
                         sizeof(float)),
        det_cls1_dim2(INPUT_H/stride[1]),
        det_cls1_dim3(INPUT_W/stride[1]),
        det_cls1_bufsize(det_cls1_dim0 * det_cls1_dim1 *
                         det_cls1_dim2 * det_cls1_dim3 *
                         sizeof(float)),
        det_cls2_dim2(INPUT_H/stride[2]),
        det_cls2_dim3(INPUT_W/stride[2]),
        det_cls2_bufsize(det_cls2_dim0 * det_cls2_dim1 *
                         det_cls2_dim2 * det_cls2_dim3 *
                         sizeof(float)),
        det_bbox0_dim2(INPUT_H/stride[0]),
        det_bbox0_dim3(INPUT_W/stride[0]),
        det_bbox0_bufsize(det_bbox0_dim0 * det_bbox0_dim1 *
                          det_bbox0_dim2 * det_bbox0_dim3 *
                          sizeof(float)),
        det_bbox1_dim2(INPUT_H/stride[1]),
        det_bbox1_dim3(INPUT_W/stride[1]),
        det_bbox1_bufsize(det_bbox1_dim0 * det_bbox1_dim1 *
                          det_bbox1_dim2 * det_bbox1_dim3 *
                          sizeof(float)),
        det_bbox2_dim2(INPUT_H/stride[2]),
        det_bbox2_dim3(INPUT_W/stride[2]),
        det_bbox2_bufsize(det_bbox2_dim0 * det_bbox2_dim1 *
                          det_bbox2_dim2 * det_bbox2_dim3 *
                          sizeof(float)),
        det_obj0_dim2(INPUT_H/stride[0]),
        det_obj0_dim3(INPUT_W/stride[0]),
        det_obj0_bufsize(det_obj0_dim0 * det_obj0_dim1 *
                         det_obj0_dim2 * det_obj0_dim3 *
                         sizeof(float)),
        det_obj1_dim2(INPUT_H/stride[1]),
        det_obj1_dim3(INPUT_W/stride[1]),
        det_obj1_bufsize(det_obj1_dim0 * det_obj1_dim1 *
                         det_obj1_dim2 * det_obj1_dim3 *
                         sizeof(float)),
        det_obj2_dim2(INPUT_H/stride[2]),
        det_obj2_dim3(INPUT_W/stride[2]),
        det_obj2_bufsize(det_obj2_dim0 * det_obj2_dim1 *
                        det_obj2_dim2 * det_obj2_dim3 *
                        sizeof(float)),
        engine_file(_engine_file)
    {
        init_context();
        std::cout<<"Inference ["<<INPUT_H<<"x"<<INPUT_W<<"] constructed"<<std::endl;
        init_done = true;
    }

    Yolo_trt::~Yolo_trt(){
        if(init_done){
            destroy_context();
            std::cout<<"Context destroyed for ["<<INPUT_H<<"x"<<INPUT_W<<"]"<<std::endl;
        }
    }

    void Yolo_trt::init_context(){
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

    void Yolo_trt::destroy_context(){
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

    int Yolo_trt::create_binding_memory(const std::string& bufname){
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

    inline void* Yolo_trt::get_infer_bufptr(const std::string& bufname, bool is_device){
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

    void Yolo_trt::do_inference(cv::Mat& image, cv::Mat& dst){
        assert(context != nullptr);

        /*  Memory address for input:focus */
        void* host_input = get_infer_bufptr(input, false /* is_device */);
        void* device_input = get_infer_bufptr(input, true /* is_device */);

        /* Memory address for output:det_cls */
        void* host_det_cls0 = get_infer_bufptr(det_cls0, false /* is_device */);
        void* device_det_cls0 = get_infer_bufptr(det_cls0, true /* is_device */);
        
        void* host_det_cls1 = get_infer_bufptr(det_cls1, false /* is_device */);
        void* device_det_cls1 = get_infer_bufptr(det_cls1, true /* is_device */);
        
        void* host_det_cls2 = get_infer_bufptr(det_cls2, false /* is_device */);
        void* device_det_cls2 = get_infer_bufptr(det_cls2, true /* is_device */);

        /* Memory address for output:det_bbox */
        void* host_det_bbox0 = get_infer_bufptr(det_bbox0, false /* is_device */);
        void* device_det_bbox0 = get_infer_bufptr(det_bbox0, true /* is_device */);
        
        void* host_det_bbox1 = get_infer_bufptr(det_bbox1, false /* is_device */);
        void* device_det_bbox1 = get_infer_bufptr(det_bbox1, true /* is_device */);
        
        void* host_det_bbox2 = get_infer_bufptr(det_bbox2, false /* is_device */);
        void* device_det_bbox2 = get_infer_bufptr(det_bbox2, true /* is_device */);

        /* Memory address for output:det_obj */
        void* host_det_obj0 = get_infer_bufptr(det_obj0, false /* is_device */);
        void* device_det_obj0 = get_infer_bufptr(det_obj0, true /* is_device */);
        
        void* host_det_obj1 = get_infer_bufptr(det_obj1, false /* is_device */);
        void* device_det_obj1 = get_infer_bufptr(det_obj1, true /* is_device */);
        
        void* host_det_obj2 = get_infer_bufptr(det_obj2, false /* is_device */);
        void* device_det_obj2 = get_infer_bufptr(det_obj2, true /* is_device */);

        float scale = std::min(INPUT_W / (image.cols*1.0), INPUT_H / (image.rows*1.0));
        pre_process(image, static_cast<float *>(host_input));

        /* upload input tensor and run inference */
        cudaMemcpyAsync(device_input, host_input, input_bufsize,
                        cudaMemcpyHostToDevice, stream);
        std::cout<<"Pre-process done!"<<std::endl;   

        bool res_ok = true;
        auto t_start1 = std::chrono::high_resolution_clock::now();

        /* Debug device_input on cuda kernel */
        context->enqueue(batchsize, &cudaOutputBuffer[0], stream, nullptr);

        auto t_end1 = std::chrono::high_resolution_clock::now();
        float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
        std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

        cudaMemcpyAsync(host_det_cls0, device_det_cls0, det_cls0_bufsize,
                    cudaMemcpyDeviceToHost, stream);
    
        cudaMemcpyAsync(host_det_cls1, device_det_cls1, det_cls1_bufsize,
                        cudaMemcpyDeviceToHost, stream);
        
        cudaMemcpyAsync(host_det_cls2, device_det_cls2, det_cls2_bufsize,
                        cudaMemcpyDeviceToHost, stream);

        cudaMemcpyAsync(host_det_bbox0, device_det_bbox0, det_bbox0_bufsize,
                        cudaMemcpyDeviceToHost, stream);
        
        cudaMemcpyAsync(host_det_bbox1, device_det_bbox1, det_bbox1_bufsize,
                        cudaMemcpyDeviceToHost, stream);
        
        cudaMemcpyAsync(host_det_bbox2, device_det_bbox2, det_bbox2_bufsize,
                        cudaMemcpyDeviceToHost, stream);

        cudaMemcpyAsync(host_det_obj0, device_det_obj0, det_obj0_bufsize,
                        cudaMemcpyDeviceToHost, stream);
        
        cudaMemcpyAsync(host_det_obj1, device_det_obj1, det_obj1_bufsize,
                        cudaMemcpyDeviceToHost, stream);
        
        cudaMemcpyAsync(host_det_obj2, device_det_obj2, det_obj2_bufsize,
                        cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERROR(res_ok);

        assert(res_ok);
        std::cout<<"Model enqueue done!"<<std::endl;
        
        /* Do post-process */
        post_process(static_cast<float *>(host_det_cls0),
                     static_cast<float *>(host_det_cls1),
                     static_cast<float *>(host_det_cls2), 
                     static_cast<float *>(host_det_bbox0), 
                     static_cast<float *>(host_det_bbox1),
                     static_cast<float *>(host_det_bbox2),
                     static_cast<float *>(host_det_obj0),
                     static_cast<float *>(host_det_obj1),
                     static_cast<float *>(host_det_obj2),
                     scale, dst);
        std::cout<<"Post-process done!"<<std::endl;
    }

    void Yolo_trt::pre_process(cv::Mat& img, float* blob){
        float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows + 0.5;
        std::cout<<"unpad_w:"<<unpad_w<<","<<"unpad_h:"<<unpad_h<<std::endl;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());
        cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 0, 0));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        std::cout<<"preimg_w:"<<out.cols<<","<<"preimg_h:"<<out.rows<<std::endl;

        cv::cvtColor(out, img, cv::COLOR_BGR2RGB);
        // float* blob = new float[img.total()*3];
        int channels = 3;
        int img_h = img.rows;
        int img_w = img.cols;
        std::cout<<channels<<","<<img_h<<","<<img_w<<","<<img.total()*3<<std::endl;
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
    }

    void Yolo_trt::post_process(float* host_det_cls0, float* host_det_cls1, float* host_det_cls2,
                             float* host_det_bbox0, float* host_det_bbox1, float* host_det_bbox2, 
                             float* host_det_obj0, float* host_det_obj1, float* host_det_obj2,
                             float r, cv::Mat& out_img){
        /* Decode detections */
        std::vector<float*> cls_vec = {host_det_cls0, host_det_cls1, host_det_cls2};
        std::vector<float*> bbox_vec = {host_det_bbox0, host_det_bbox1, host_det_bbox2};
        std::vector<float*> obj_vec = {host_det_obj0, host_det_obj1, host_det_obj2};
        std::vector<Object> det_objs;
        for(int n=0; n<3; n++) // strides
        {
            int num_grid_x = (int) (input_dim2) / stride[n];
            int num_grid_y = (int) (input_dim3) / stride[n];
            float* cls_buffer = cls_vec[n];
            float* bbox_buffer = bbox_vec[n];
            float* obj_buffer = obj_vec[n];
            

            for(int i=0; i<num_grid_y; i++) // grids
            {
                for(int j=0; j<num_grid_x; j++)
                {
                    float obj = obj_buffer[i * num_grid_x + j];
                    obj = 1 / (1 + expf(-obj));
                    
                    if(obj < 0.1) continue; // FIXME : to parameterize
                    
                    float x_feat = bbox_buffer[i * num_grid_x + j];
                    float y_feat = bbox_buffer[num_grid_y * num_grid_x + (i * num_grid_x + j)];
                    float w_feat = bbox_buffer[num_grid_y * num_grid_x * 2 + (i * num_grid_x + j)];
                    float h_feat = bbox_buffer[num_grid_y * num_grid_x * 3 + (i * num_grid_x + j)];
                    
                    float x_center = (x_feat + j) * stride[n];
                    float y_center = (y_feat + i) * stride[n];
                    float w = expf(w_feat) * stride[n];
                    float h = expf(h_feat) * stride[n];

                    for(int k=0; k<det_cls0_dim1; k++)
                    {
                        float cls = cls_buffer[k * num_grid_x * num_grid_y + (i * num_grid_x + j)];
                        cls = 1 / (1 + expf(-cls));
                        float score = cls * obj;
                        
                        if(score > confThreshold)
                        {
                            int left = (x_center - 0.5 * w) / r;
                            int top = (y_center - 0.5 * h ) / r;
                            int ww = (int)(w / r);
                            int hh = (int)(h / r);

                            int right = left + ww;
                            int bottom = top + hh;
                            
                            /* clip */
                            left = std::min(std::max(0, left), out_img.cols);
                            top = std::min(std::max(0, top), out_img.rows);
                            right = std::min(std::max(0, right), out_img.cols);
                            bottom = std::min(std::max(0, bottom), out_img.rows);
                                
                            Object obj;
                            obj.rect = cv::Rect_<float>(left, top, right - left,  bottom - top);
                            obj.label = 0;
                            obj.prob = score;
                            det_objs.emplace_back(obj);
                        }
                        
                    }

                }
            }
            
        }
        
        /* Perform non maximum suppression */
        std::vector<int> picked;
        nms_sorted_bboxes(det_objs, picked, nmsThreshold);
        
        int count = picked.size();
        std::cout<<"Num of boxes: "<< count <<std::endl;
        
        /* Draw & show res */
        for(int i=0; i<count; ++i)
        {
            const Object& obj = det_objs[picked[i]];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = x1 + obj.rect.width;
            float y2 = y1 + obj.rect.height;
            
            cv::rectangle(out_img, cv::Point(x1, y1), cv::Point(x2, y2), 
                        cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
}//namespace yolox