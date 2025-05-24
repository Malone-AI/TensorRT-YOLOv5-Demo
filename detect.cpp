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
    std::vector<size_t> tensor_sizes;
    cudaStream_t stream;
    
    // 模型参数
    int input_size = 640;
    int num_classes = 80;
    bool initialized = false;
    
    // 输出相关
    int main_output_idx = -1;  // 主输出张量索引
    
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
        
        std::vector<char> modelStream(size);
        file.read(modelStream.data(), size);
        file.close();
        
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(modelStream.data(), size));
        context.reset(engine->createExecutionContext());
        
        return engine && context;
    }
    
    bool analyzeEngine() {
        std::cout << "\n=== 引擎分析 ===" << std::endl;
        
        int num_io_tensors = engine->getNbIOTensors();
        std::cout << "输入输出张量数量: " << num_io_tensors << std::endl;
        
        buffers.resize(num_io_tensors);
        tensor_names.resize(num_io_tensors);
        tensor_sizes.resize(num_io_tensors);
        
        for (int i = 0; i < num_io_tensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            tensor_names[i] = std::string(name);
            nvinfer1::Dims dims = engine->getTensorShape(name);
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
            tensor_sizes[i] = tensor_size;
            
            // 找到主输出张量
            if (!is_input) {
                if (std::string(name) == "output" || 
                    (dims.nbDims == 3 && dims.d[1] == 25200 && dims.d[2] == 85)) {
                    main_output_idx = i;
                    std::cout << "  >>> 这是主输出张量 <<<" << std::endl;
                }
            }
            
            std::cout << "  字节大小: " << tensor_size << std::endl;
        }
        
        if (main_output_idx == -1) {
            std::cerr << "警告: 未找到主输出张量，使用第一个输出" << std::endl;
            // 第一个输出张量
            for (int i = 0; i < num_io_tensors; ++i) {
                const char* name = tensor_names[i].c_str();
                if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                    main_output_idx = i;
                    break;
                }
            }
        }
        
        std::cout << "使用输出张量索引: " << main_output_idx << std::endl;
        std::cout << "=== 分析完成 ===\n" << std::endl;
        return true;
    }
    
    bool allocateBuffers() {
        std::cout << "分配GPU内存..." << std::endl;
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            cudaError_t err = cudaMalloc(&buffers[i], tensor_sizes[i]);
            if (err != cudaSuccess) {
                std::cerr << "分配GPU内存失败 [" << tensor_names[i] << "]: " 
                          << cudaGetErrorString(err) << std::endl;
                return false;
            }
            
            const char* name = tensor_names[i].c_str();
            bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            std::cout << "  " << (is_input ? "输入" : "输出") 
                      << " [" << tensor_names[i] << "]: " << tensor_sizes[i] << " 字节" << std::endl;
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
        if (!initialized || main_output_idx == -1) {
            std::cerr << "检测器未正确初始化" << std::endl;
            return {};
        }
        
        float x_scale, y_scale;
        cv::Mat processed = preprocess(image, x_scale, y_scale);
        
        // 准备输入数据
        std::vector<float> input_data(input_size * input_size * 3);
        
        // 填充数据 - NCHW格式
        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_size; ++h) {
                for (int w = 0; w < input_size; ++w) {
                    input_data[idx++] = processed.at<cv::Vec3f>(h, w)[2-c]; // BGR to RGB
                }
            }
        }
        
        // 找到输入张量索引
        int input_idx = -1;
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const char* name = tensor_names[i].c_str();
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_idx = i;
                break;
            }
        }
        
        if (input_idx == -1) {
            std::cerr << "未找到输入张量" << std::endl;
            return {};
        }
        
        // 复制数据到GPU
        cudaError_t err = cudaMemcpyAsync(buffers[input_idx], input_data.data(), 
                                         tensor_sizes[input_idx], 
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
        
        // 获取主输出数据
        std::vector<float> output_data(tensor_sizes[main_output_idx] / sizeof(float));
        
        err = cudaMemcpyAsync(output_data.data(), buffers[main_output_idx], 
                             tensor_sizes[main_output_idx], 
                             cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            std::cerr << "数据从GPU复制失败: " << cudaGetErrorString(err) << std::endl;
            return {};
        }
        
        cudaStreamSynchronize(stream);
        
        // 调试输出
        std::cout << "输出数据大小: " << output_data.size() << std::endl;
        
        // 检查前几个输出值
        if (output_data.size() >= 10) {
            std::cout << "前10个输出值: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << output_data[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // 后处理
        return postprocess(output_data, x_scale, y_scale);
    }
    
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      float x_scale, float y_scale) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        // YOLOv5输出格式：[1, 25200, 85] 其中85 = 4(坐标) + 1(置信度) + 80(类别)
        int num_detections = 25200;
        int num_features = 85;
        
        // 确保输出数据大小正确
        if (output.size() < num_detections * num_features) {
            std::cerr << "输出数据大小不匹配: " << output.size() 
                      << " < " << (num_detections * num_features) << std::endl;
            return {};
        }
        
        std::cout << "开始后处理，检测数量: " << num_detections << std::endl;
        
        int valid_detections = 0;
        for (int i = 0; i < num_detections; ++i) {
            int offset = i * num_features;
            
            // YOLOv5格式：cx, cy, w, h, confidence, class_probs...
            float cx = output[offset + 0];
            float cy = output[offset + 1];
            float w = output[offset + 2];
            float h = output[offset + 3];
            float confidence = output[offset + 4];
            
            if (confidence < 0.25) continue; // 置信度阈值
            
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
            if (final_confidence < 0.3) continue; // 最终置信度阈值
            
            valid_detections++;
            
            // 转换坐标（从中心点到左上角）
            float x = (cx - w / 2) * x_scale;
            float y = (cy - h / 2) * y_scale;
            w *= x_scale;
            h *= y_scale;
            
            // 确保边界框有效
            if (x >= 0 && y >= 0 && w > 0 && h > 0) {
                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(final_confidence);
                class_ids.push_back(class_id);
            }
        }
        
        std::cout << "有效检测数量: " << valid_detections << std::endl;
        std::cout << "通过边界检查的检测数量: " << boxes.size() << std::endl;
        
        // NMS
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.3, 0.4, indices);
            std::cout << "NMS后的检测数量: " << indices.size() << std::endl;
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
    
    if (detections.size() > 0) {
        detector.drawDetections(image, detections);
        
        // 保存结果
        std::string output_path = "detection_result.jpg";
        cv::imwrite(output_path, image);
        std::cout << "检测结果已保存到: " << output_path << std::endl;
    }
    
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
    
    // 获取视频信息
    int fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "视频信息: " << width << "x" << height << " @ " << fps << "fps" << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    int success_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 先测试一帧
    cap >> frame;
    if (!frame.empty()) {
        std::cout << "测试第一帧..." << std::endl;
        std::vector<Detection> test_detections = detector.detect(frame);
        std::cout << "第一帧测试完成，检测到 " << test_detections.size() << " 个目标" << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 重置到开始
    }
    
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
        
        // 每10帧输出一次状态
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