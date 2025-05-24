#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[" << static_cast<int>(severity) << "] " << msg << std::endl;
        }
    }
} gLogger;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class YOLOv5TensorRT {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    std::vector<void*> buffers;
    std::vector<std::string> tensor_names;
    cudaStream_t stream;
    
    // 动态获取的参数
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;
    int input_size = 640;
    int num_classes = 80;
    int output_size = 25200;
    size_t input_size_bytes = 0;
    size_t output_size_bytes = 0;
    bool initialized = false;
    
    // COCO类别名称
    std::vector<std::string> class_names = {
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
    
public:
    YOLOv5TensorRT(const std::string& engine_path) {
        if (!loadEngine(engine_path)) {
            std::cerr << "引擎加载失败" << std::endl;
            return;
        }
        
        if (!analyzeEngine()) {
            std::cerr << "引擎分析失败" << std::endl;
            return;
        }
        
        if (!allocateBuffers()) {
            std::cerr << "内存分配失败" << std::endl;
            return;
        }
        
        cudaStreamCreate(&stream);
        initialized = true;
        
        std::cout << "YOLOv5 TensorRT 初始化成功" << std::endl;
    }
    
    ~YOLOv5TensorRT() {
        cleanup();
    }
    
    bool isInitialized() const { return initialized; }
    
private:
    bool loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
            return false;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        if (size == 0) {
            std::cerr << "引擎文件为空" << std::endl;
            return false;
        }
        
        std::vector<char> modelStream(size);
        file.read(modelStream.data(), size);
        file.close();
        
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            std::cerr << "创建runtime失败" << std::endl;
            return false;
        }
        
        engine.reset(runtime->deserializeCudaEngine(modelStream.data(), size));
        if (!engine) {
            std::cerr << "反序列化引擎失败" << std::endl;
            return false;
        }
        
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "创建execution context失败" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool analyzeEngine() {
        std::cout << "\n=== 引擎分析 ===" << std::endl;
        
        int num_io_tensors = engine->getNbIOTensors();
        std::cout << "输入输出张量数量: " << num_io_tensors << std::endl;
        
        buffers.resize(num_io_tensors);
        tensor_names.resize(num_io_tensors);
        
        // 分析所有输入输出张量
        for (int i = 0; i < num_io_tensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            tensor_names[i] = std::string(name);
            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::DataType dtype = engine->getTensorDataType(name);
            bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            
            std::cout << "张量 " << i << ": " << name 
                      << " [" << (is_input ? "输入" : "输出") << "]" << std::endl;
            std::cout << "  形状: ";
            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << dims.d[j];
                if (j < dims.nbDims - 1) std::cout << "x";
            }
            std::cout << std::endl;
            
            // 计算张量大小
            size_t tensor_size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                tensor_size *= dims.d[j];
            }
            tensor_size *= sizeof(float);
            
            if (is_input) {
                input_dims = dims;
                if (dims.nbDims == 4) { // NCHW
                    input_size = dims.d[2]; // H
                }
                input_size_bytes = tensor_size;
                std::cout << "  输入尺寸: " << input_size << "x" << input_size << std::endl;
                std::cout << "  输入字节大小: " << input_size_bytes << std::endl;
            } else {
                output_dims = dims;
                output_size_bytes = tensor_size;
                
                // 根据输出维度推断格式
                if (dims.nbDims == 3) { // [1, num_detections, 85]
                    output_size = dims.d[1];
                    num_classes = dims.d[2] - 5;
                } else if (dims.nbDims == 2) { // [num_detections, 85]
                    output_size = dims.d[0];
                    num_classes = dims.d[1] - 5;
                }
                
                std::cout << "  输出数量: " << output_size << std::endl;
                std::cout << "  类别数量: " << num_classes << std::endl;
                std::cout << "  输出字节大小: " << output_size_bytes << std::endl;
            }
        }
        
        std::cout << "=== 分析完成 ===\n" << std::endl;
        return true;
    }
    
    bool allocateBuffers() {
        std::cout << "分配GPU内存..." << std::endl;
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            const char* name = tensor_names[i].c_str();
            bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            
            size_t size = is_input ? input_size_bytes : output_size_bytes;
            
            cudaError_t err = cudaMalloc(&buffers[i], size);
            if (err != cudaSuccess) {
                std::cerr << "分配GPU内存失败 [" << name << "]: " 
                          << cudaGetErrorString(err) << std::endl;
                // 清理已分配的内存
                for (size_t j = 0; j < i; ++j) {
                    if (buffers[j]) cudaFree(buffers[j]);
                }
                return false;
            }
            
            std::cout << "  " << (is_input ? "输入" : "输出") 
                      << " [" << name << "]: " << size << " 字节" << std::endl;
        }
        
        std::cout << "GPU内存分配成功" << std::endl;
        return true;
    }
    
    void cleanup() {
        for (void* buffer : buffers) {
            if (buffer) cudaFree(buffer);
        }
        buffers.clear();
        
        if (stream) { 
            cudaStreamDestroy(stream); 
            stream = nullptr; 
        }
    }
    
public:
    cv::Mat preprocess(const cv::Mat& image, float& x_scale, float& y_scale) {
        x_scale = (float)image.cols / input_size;
        y_scale = (float)image.rows / input_size;
        
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_size, input_size));
        
        cv::Mat float_image;
        resized.convertTo(float_image, CV_32F, 1.0/255.0);
        
        return float_image;
    }
    
    std::vector<Detection> detect(const cv::Mat& image) {
        if (!initialized) {
            std::cerr << "检测器未初始化" << std::endl;
            return {};
        }
        
        float x_scale, y_scale;
        cv::Mat processed = preprocess(image, x_scale, y_scale);
        
        // 准备输入数据
        std::vector<float> input_data(input_size_bytes / sizeof(float));
        
        // 填充数据 - NCHW格式
        if (input_dims.nbDims == 4) {
            int idx = 0;
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < input_size; ++h) {
                    for (int w = 0; w < input_size; ++w) {
                        input_data[idx++] = processed.at<cv::Vec3f>(h, w)[2-c]; // BGR to RGB
                    }
                }
            }
        } else {
            std::cerr << "不支持的输入维度格式" << std::endl;
            return {};
        }
        
        // 复制数据到GPU
        cudaError_t err = cudaMemcpyAsync(buffers[0], input_data.data(), 
                                         input_size_bytes, 
                                         cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            std::cerr << "数据复制到GPU失败: " << cudaGetErrorString(err) << std::endl;
            return {};
        }
        
        // 设置所有张量地址
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const char* name = tensor_names[i].c_str();
            if (!context->setTensorAddress(name, buffers[i])) {
                std::cerr << "设置张量地址失败: " << name << std::endl;
                return {};
            }
        }
        
        // 执行推理
        if (!context->enqueueV3(stream)) {
            std::cerr << "推理执行失败" << std::endl;
            return {};
        }
        
        // 获取输出数据
        std::vector<float> output_data(output_size_bytes / sizeof(float));
        
        // 找到输出缓冲区（通常是最后一个）
        void* output_buffer = nullptr;
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const char* name = tensor_names[i].c_str();
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                output_buffer = buffers[i];
                break;
            }
        }
        
        if (!output_buffer) {
            std::cerr << "未找到输出缓冲区" << std::endl;
            return {};
        }
        
        err = cudaMemcpyAsync(output_data.data(), output_buffer, 
                             output_size_bytes, 
                             cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            std::cerr << "数据从GPU复制失败: " << cudaGetErrorString(err) << std::endl;
            return {};
        }
        
        cudaStreamSynchronize(stream);
        
        // 后处理
        return postprocess(output_data, x_scale, y_scale);
    }
    
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      float x_scale, float y_scale) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        int stride = num_classes + 5; // x,y,w,h,conf + classes
        
        for (int i = 0; i < output_size; ++i) {
            int offset = i * stride;
            
            if (offset + stride > (int)output.size()) {
                break;
            }
            
            float confidence = output[offset + 4];
            if (confidence < 0.25) continue; // 降低阈值
            
            // 找到最大类别概率
            float max_class_score = 0;
            int class_id = 0;
            for (int j = 0; j < num_classes; ++j) {
                float score = output[offset + 5 + j];
                if (score > max_class_score) {
                    max_class_score = score;
                    class_id = j;
                }
            }
            
            float final_confidence = confidence * max_class_score;
            if (final_confidence < 0.3) continue; // 降低阈值
            
            float x = output[offset + 0] * x_scale;
            float y = output[offset + 1] * y_scale;
            float w = output[offset + 2] * x_scale;
            float h = output[offset + 3] * y_scale;
            
            int left = x - w / 2;
            int top = y - h / 2;
            
            // 确保边界框在图像范围内
            if (left >= 0 && top >= 0 && w > 0 && h > 0) {
                boxes.push_back(cv::Rect(left, top, w, h));
                confidences.push_back(final_confidence);
                class_ids.push_back(class_id);
            }
        }
        
        // NMS
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.3, 0.4, indices);
        }
        
        std::vector<Detection> detections;
        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx];
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            detections.push_back(det);
        }
        
        return detections;
    }
    
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            
            std::string label = (det.class_id < (int)class_names.size()) ? 
                               class_names[det.class_id] : ("class_" + std::to_string(det.class_id));
            label += " " + std::to_string(det.confidence).substr(0, 4);
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            cv::rectangle(image, 
                         cv::Point(det.box.x, det.box.y - labelSize.height - 10),
                         cv::Point(det.box.x + labelSize.width, det.box.y),
                         cv::Scalar(0, 255, 0), -1);
            
            cv::putText(image, label, 
                       cv::Point(det.box.x, det.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
};

void processImage(YOLOv5TensorRT& detector, const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << image_path << std::endl;
        return;
    }
    
    std::cout << "处理图像: " << image_path << std::endl;
    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections = detector.detect(image);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "检测到 " << detections.size() << " 个目标，推理时间: " << duration.count() << "ms" << std::endl;
    
    detector.drawDetections(image, detections);
    
    cv::imshow("YOLOv5 Detection", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void processVideo(YOLOv5TensorRT& detector, const std::string& video_path) {
    cv::VideoCapture cap;
    
    if (video_path == "0") {
        cap.open(0);
        std::cout << "打开摄像头进行实时检测" << std::endl;
    } else {
        cap.open(video_path);
        std::cout << "处理视频: " << video_path << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频源: " << video_path << std::endl;
        return;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    int success_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        frame_count++;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections = detector.detect(frame);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (detections.size() > 0) {
            success_count++;
            detector.drawDetections(frame, detections);
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "帧 " << frame_count << ": " << detections.size() 
                      << " 个目标, " << duration.count() << "ms" << std::endl;
        }
        
        // 显示信息
        std::string info = "Frame: " + std::to_string(frame_count) + 
                          " | Objects: " + std::to_string(detections.size()) +
                          " | Success: " + std::to_string(success_count);
        
        cv::putText(frame, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("YOLOv5 Video Detection", frame);
        
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        
        // 限制处理速度以便观察
        if (frame_count % 10 == 0) {
            std::cout << "已处理 " << frame_count << " 帧，检测成功 " << success_count << " 帧" << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\n处理完成!" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "成功检测帧数: " << success_count << std::endl;
    std::cout << "成功率: " << (100.0 * success_count / frame_count) << "%" << std::endl;
    std::cout << "总时间: " << total_duration.count() << "ms" << std::endl;
    std::cout << "平均FPS: " << (frame_count * 1000.0 / total_duration.count()) << std::endl;
    
    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法:" << std::endl;
        std::cout << "  图片推理: " << argv[0] << " <engine_file> <image_file>" << std::endl;
        std::cout << "  视频推理: " << argv[0] << " <engine_file> <video_file>" << std::endl;
        std::cout << "  摄像头推理: " << argv[0] << " <engine_file> 0" << std::endl;
        return -1;
    }
    
    std::string engine_path = argv[1];
    std::string input_path = argv[2];
    
    // 初始化检测器
    YOLOv5TensorRT detector(engine_path);
    
    if (!detector.isInitialized()) {
        std::cerr << "检测器初始化失败" << std::endl;
        return -1;
    }
    
    // 判断输入类型
    if (input_path == "0" || input_path.find(".mp4") != std::string::npos || 
        input_path.find(".avi") != std::string::npos || input_path.find(".mov") != std::string::npos) {
        processVideo(detector, input_path);
    } else {
        processImage(detector, input_path);
    }
    
    return 0;
}