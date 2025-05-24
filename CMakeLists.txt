cmake_minimum_required(VERSION 3.10)
project(YOLOv5_Inference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 查找必要的包
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/include/x86_64-linux-gnu)

# 可执行文件
add_executable(detect detect.cpp)

# 链接库
target_link_libraries(detect 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    cudart
)