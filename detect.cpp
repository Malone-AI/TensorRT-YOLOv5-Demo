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
    
    int input_size = 640;
    int num_classes = 80;
    bool initialized = false;
    int main_output_idx = -1;
    
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
        if (!file.good()) return false;
        
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
        int num_io_tensors = engine->getNbIOTensors();
        
        buffers.resize(num_io_tensors);
        tensor_names.resize(num_io_tensors);
        tensor_sizes.resize(num_io_tensors);
        
        for (int i = 0; i < num_io_tensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            tensor_names[i] = std::string(name);
            nvinfer1::Dims dims = engine->getTensorShape(name);
            bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            
            size_t tensor_size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                tensor_size *= dims.d[j];
            }
            tensor_size *= sizeof(float);
            tensor_sizes[i] = tensor_size;
            
            // 找到主输出张量
            if (!is_input && (std::string(name) == "output" || 
                (dims.nbDims == 3 && dims.d[1] == 25200 && dims.d[2] == 85))) {
                main_output_idx = i;
            }
        }
        
        // 如果没找到主输出，使用第一个输出
        if (main_output_idx == -1) {
            for (int i = 0; i < num_io_tensors; ++i) {
                const char* name = tensor_names[i].c_str();
                if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                    main_output_idx = i;
                    break;
                }
            }
        }
        
        return main_output_idx != -1;
    }
    
    bool allocateBuffers() {
        for (size_t i = 0; i < buffers.size(); ++i) {
            cudaError_t err = cudaMalloc(&buffers[i], tensor_sizes[i]);
            if (err != cudaSuccess) return false;
        }
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
            return {};
        }
        
        float x_scale, y_scale;
        cv::Mat processed = preprocess(image, x_scale, y_scale);
        
        // 准备输入数据
        std::vector<float> input_data(input_size * input_size * 3);
        
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
        
        if (input_idx == -1) return {};
        
        // 复制数据到GPU
        if (cudaMemcpyAsync(buffers[input_idx], input_data.data(), 
                           tensor_sizes[input_idx], 
                           cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return {};
        }
        
        // 设置张量地址
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const char* name = tensor_names[i].c_str();
            context->setTensorAddress(name, buffers[i]);
        }
        
        // 执行推理
        if (!context->enqueueV3(stream)) {
            return {};
        }
        
        // 获取输出数据
        std::vector<float> output_data(tensor_sizes[main_output_idx] / sizeof(float));
        
        if (cudaMemcpyAsync(output_data.data(), buffers[main_output_idx], 
                           tensor_sizes[main_output_idx], 
                           cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            return {};
        }
        
        cudaStreamSynchronize(stream);
        
        return postprocess(output_data, x_scale, y_scale);
    }
    
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      float x_scale, float y_scale) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        int num_detections = 25200;
        int num_features = 85;
        
        if (output.size() < num_detections * num_features) {
            return {};
        }
        
        for (int i = 0; i < num_detections; ++i) {
            int offset = i * num_features;
            
            float cx = output[offset + 0];
            float cy = output[offset + 1];
            float w = output[offset + 2];
            float h = output[offset + 3];
            float confidence = output[offset + 4];
            
            if (confidence < 0.4) continue;
            
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
            if (final_confidence < 0.5) continue;
            
            // 转换坐标
            float x = (cx - w / 2) * x_scale;
            float y = (cy - h / 2) * y_scale;
            w *= x_scale;
            h *= y_scale;
            
            if (x >= 0 && y >= 0 && w > 0 && h > 0) {
                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(final_confidence);
                class_ids.push_back(class_id);
            }
        }
        
        // NMS
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
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
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections = detector.detect(image);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "检测到 " << detections.size() << " 个目标，推理时间: " << duration.count() << "ms" << std::endl;
    
    if (detections.size() > 0) {
        detector.drawDetections(image, detections);
        cv::imwrite("detection_result.jpg", image);
        std::cout << "检测结果已保存到: detection_result.jpg" << std::endl;
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
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    std::cout << "视频信息: " << width << "x" << height << " @ " << fps << "fps, 总帧数: " << total_frames << std::endl;
    
    // 创建输出视频writer
    cv::VideoWriter output_video("output_detection.mp4", 
                                cv::VideoWriter::fourcc('m','p','4','v'), 
                                fps, cv::Size(width, height));
    
    if (!output_video.isOpened()) {
        std::cout << "警告: 无法创建输出视频文件，将只显示实时结果" << std::endl;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    int success_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "视频处理完成或帧为空" << std::endl;
            break;
        }
        
        frame_count++;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections = detector.detect(frame);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (detections.size() > 0) {
            success_count++;
            detector.drawDetections(frame, detections);
            std::cout << "帧 " << frame_count << "/" << total_frames 
                      << ": 检测到 " << detections.size() << " 个目标 (" 
                      << duration.count() << "ms)" << std::endl;
        }
        
        // 显示帧信息
        std::string info = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames) +
                          " | Objects: " + std::to_string(detections.size()) +
                          " | Time: " + std::to_string(duration.count()) + "ms";
        
        cv::putText(frame, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // 写入输出视频
        if (output_video.isOpened()) {
            output_video.write(frame);
        }
        
        // 显示当前帧
        cv::imshow("YOLOv5 Video Detection", frame);
        
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            std::cout << "用户中断处理" << std::endl;
            break;
        }
        
        // 每50帧输出一次进度
        if (frame_count % 50 == 0) {
            float progress = (float)frame_count / total_frames * 100;
            std::cout << "进度: " << progress << "% (" << frame_count << "/" << total_frames 
                      << "), 成功检测: " << success_count << " 帧" << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\n=== 处理完成 ===" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "成功检测帧数: " << success_count << std::endl;
    std::cout << "成功率: " << (frame_count > 0 ? 100.0 * success_count / frame_count : 0) << "%" << std::endl;
    std::cout << "总处理时间: " << total_duration.count() << "ms" << std::endl;
    std::cout << "平均FPS: " << (frame_count * 1000.0 / total_duration.count()) << std::endl;
    
    if (output_video.isOpened()) {
        std::cout << "检测结果视频已保存到: output_detection.mp4" << std::endl;
        output_video.release();
    }
    
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
    
    YOLOv5TensorRT detector(engine_path);
    
    if (!detector.isInitialized()) {
        std::cerr << "检测器初始化失败" << std::endl;
        return -1;
    }
    
    if (input_path == "0" || input_path.find(".mp4") != std::string::npos || 
        input_path.find(".avi") != std::string::npos || input_path.find(".mov") != std::string::npos) {
        processVideo(detector, input_path);
    } else {
        processImage(detector, input_path);
    }
    
    return 0;
}