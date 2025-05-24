import tensorrt as trt
import torch
import numpy as np
from pathlib import Path
import sys
import argparse

class Logger(trt.ILogger):
    def log(self, severity, msg):
        if severity <= trt.Logger.WARNING:
            print(f"[{severity}] {msg}")

def onnx_to_engine(onnx_path, engine_path, precision='fp16', workspace=4):
    """将ONNX模型转换为TensorRT引擎"""
    logger = Logger()
    
    # 创建builder
    builder = trt.Builder(logger)
    
    # 设置网络配置
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX文件
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # workspace GB
    
    # 设置精度
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
    else:
        print("Using FP32 precision")
    
    # 构建引擎
    print(f"Building TensorRT engine from {onnx_path}...")
    print("This may take several minutes...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build the engine")
        return None
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved to {engine_path}")
    return serialized_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str)
    parser.add_argument("--output", type = str)
    args = parser.parse_args()
    if not args.output:
        args.output = str(Path(args.input).stem) + ".engine"
    onnx_to_engine(args.input, args.output)