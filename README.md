# YOLOv5 TensorRT 推理演示

这是一个基于TensorRT的YOLOv5目标检测推理项目，支持图像和视频的实时目标检测。

## 功能特性

- 🚀 **高性能推理**: 使用TensorRT优化，支持FP16精度，大幅提升推理速度
- 🎯 **多格式支持**: 支持图像、视频文件和实时摄像头检测
- 📦 **完整工作流**: 从ONNX模型转换到TensorRT引擎，再到推理检测
- 🎨 **可视化输出**: 实时显示检测结果，支持保存输出视频
- 📊 **性能统计**: 显示FPS、检测成功率等性能指标

## 环境要求

- CUDA 12.8
- TensorRT 10.0
- OpenCV 4
- CMake 3.10+
- g++

### Python依赖（用于模型转换）
```bash
pip install tensorrt torch numpy
```

## 项目结构

```
demo/
├── CMakeLists.txt          # CMake构建配置
├── convert_to_engine.py    # ONNX到TensorRT引擎转换脚本
├── detect.cpp              # C++推理代码
├── yolov5s.onnx           # YOLOv5 ONNX模型文件
├── yolov5s.engine         # TensorRT引擎文件
└── README.md              # 本文档
```

## 快速开始

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd tensorrtDemo/code_for_student/demo
```

### 2. 模型转换（可选）
推荐通过 yolov5 官方项目中的 export.py 导出ONNX模型:
```bash
python export.py --weights yolov5s.pt
```

之后会在统一目录生成 yolov5s.onnx文件。

使用该 ONNX 模型或者自己的 ONNX 模型，转换为TensorRT引擎：

```bash
# 转换为FP16精度引擎（推荐）
python convert_to_engine.py --input yolov5s.onnx --output yolov5s.engine --precision fp16

# 转换为FP32精度引擎
python convert_to_engine.py --input yolov5s.onnx --output yolov5s.engine --precision fp32

# 转换为INT8精度引擎（需要校准数据）
python convert_to_engine.py --input yolov5s.onnx --output yolov5s.engine --precision int8
```

### 3. 编译项目
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 4. 运行推理

#### 图像检测
```bash
# 检测单张图像
./detect ../yolov5s.engine path/to/your/image.jpg
```

#### 视频检测
```bash
# 检测视频文件
./detect ../yolov5s.engine path/to/your/video.mp4
```

#### 实时摄像头检测
```bash
# 使用默认摄像头
./detect ../yolov5s.engine 0
```

## 使用示例

### 基本用法
```bash
# 检测图片
./detect yolov5s.engine test_image.jpg

# 检测视频
./detect yolov5s.engine test_video.mp4

# 实时检测，调用摄像头
./detect yolov5s.engine 0
```

### 输出文件
- **图像检测**: 保存检测结果到 `detection_result.jpg`
- **视频检测**: 保存检测结果视频到 `output_detection.mp4`
- **实时检测**: 仅显示，不保存文件

## 性能调优

### 1. 精度设置
- **FP32**: 最高精度，速度较慢
- **FP16**: 平衡精度和速度，推荐使用
- **INT8**: 最快速度，需要校准数据

### 2. 检测参数调整
在 `detect.cpp` 中可以调整以下参数：
```cpp
// 置信度阈值
if (confidence < 0.4) continue;  // 降低可检测更多目标

// NMS阈值
cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
```

### 3. 输入尺寸
- 默认输入尺寸: 640x640
- 可以修改为其他尺寸如 416x416, 512x512, 832x832
- 更大尺寸精度更高但速度更慢

## 支持的目标类别

支持COCO数据集的80个类别：
```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, 
traffic light, fire hydrant, stop sign, parking meter, bench, bird, 
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, 
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, 
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, 
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, 
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, 
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, 
remote, keyboard, cell phone, microwave, oven, toaster, sink, 
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, 
toothbrush
```

## 故障排除

### 常见问题

1. **引擎加载失败**
   ```
   解决方案: 检查engine文件路径和权限
   ```

2. **CUDA内存不足**
   ```
   解决方案: 减小batch size或降低输入分辨率
   ```

3. **推理速度慢**
   ```
   解决方案: 
   - 使用FP16精度
   - 检查GPU利用率
   - 确保CUDA版本匹配
   ```

4. **检测效果差**
   ```
   解决方案:
   - 调整置信度阈值
   - 检查输入图像预处理
   - 验证模型转换正确性
   ```

### 查看详细日志
运行时会显示详细的推理信息：
- 引擎分析结果
- 张量形状和大小
- 检测统计信息
- 性能指标

## 扩展开发

### 添加新的后处理
在 `postprocess` 函数中修改检测逻辑：
```cpp
std::vector<Detection> postprocess(const std::vector<float>& output, 
                                  float x_scale, float y_scale) {
    // 自定义后处理逻辑
}
```

### 支持其他模型
修改以下参数以适配其他YOLO模型：
- `num_classes`: 类别数量
- `num_detections`: 检测框数量
- `input_size`: 输入尺寸

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献

欢迎提交Issue和Pull Request！

如有问题请提出issue

## 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5) - 原始模型实现
- [TensorRT](https://developer.nvidia.com/tensorrt) - 推理优化框架
- [OpenCV](https://opencv.org/) - 计算机视觉库