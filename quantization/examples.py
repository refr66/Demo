import torch
import numpy as np
import os
from typing import List, Tuple

# 导入量化工具
from .quantization import PTQQuantizer, quantize_model, quantize_tensor, dequantize_tensor
from .calibration import calibrate_model, create_minmax_calibrator
from .inference import QuantizedModel, simulate_int8_inference, run_int8_matmul, compare_quantization_accuracy


def create_dummy_model() -> torch.nn.Module:
    """
    创建一个简单的测试模型
    """
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 256)
            self.relu1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 128)
            self.relu2 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x
    
    return SimpleModel()


def generate_calibration_data(num_samples: int = 100) -> List[torch.Tensor]:
    """
    生成校准数据
    
    Args:
        num_samples: 样本数量
    
    Returns:
        校准数据列表
    """
    # 模拟MNIST数据格式 (batch_size, 784)
    return [torch.randn(32, 784) for _ in range(num_samples)]


def example_basic_quantization():
    """
    基本量化示例
    """
    print("===== 基本量化示例 =====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 保存原始模型大小
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
    
    # 量化模型
    quantized_model = quantize_model(model, calibration_data)
    
    # 计算量化后模型大小（假设权重是INT8，其他参数保持FP32）
    quantized_size = 0
    for param in quantized_model.parameters():
        if param.dtype == torch.int8:
            quantized_size += param.numel() * 1  # INT8占用1字节
        else:
            quantized_size += param.numel() * 4  # FP32占用4字节
    quantized_size = quantized_size / 1024 / 1024  # MB
    
    # 测试模型推理
    test_input = torch.randn(1, 784)
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
    
    # 计算误差
    mse = torch.nn.functional.mse_loss(quantized_output, original_output).item()
    
    print(f"原始模型大小: {original_size:.4f} MB")
    print(f"量化模型大小: {quantized_size:.4f} MB")
    print(f"压缩率: {original_size/quantized_size:.2f}x")
    print(f"输出MSE误差: {mse:.6f}")
    print()


def example_custom_quantization_params():
    """
    自定义量化参数示例
    """
    print("===== 自定义量化参数示例 =====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 创建自定义量化器
    print("使用对称量化...")
    symmetric_quantizer = PTQQuantizer(bits=8, symmetric=True)
    symmetric_quantized_model, symmetric_params = symmetric_quantizer.quantize_model(model, calibration_data)
    
    # 创建新模型用于非对称量化
    model2 = create_dummy_model()
    
    print("使用非对称量化...")
    asymmetric_quantizer = PTQQuantizer(bits=8, symmetric=False)
    asymmetric_quantized_model, asymmetric_params = asymmetric_quantizer.quantize_model(model2, calibration_data)
    
    # 测试模型推理
    test_input = torch.randn(1, 784)
    with torch.no_grad():
        original_output = model(test_input)
        symmetric_output = symmetric_quantized_model(test_input)
        asymmetric_output = asymmetric_quantized_model(test_input)
    
    # 计算误差
    symmetric_mse = torch.nn.functional.mse_loss(symmetric_output, original_output).item()
    asymmetric_mse = torch.nn.functional.mse_loss(asymmetric_output, original_output).item()
    
    print(f"对称量化MSE误差: {symmetric_mse:.6f}")
    print(f"非对称量化MSE误差: {asymmetric_mse:.6f}")
    print(f"哪种量化更好? {'非对称' if asymmetric_mse < symmetric_mse else '对称'}")
    print()


def example_different_calibration_methods():
    """
    不同校准方法示例
    """
    print("===== 不同校准方法示例 =====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 使用minmax校准
    print("使用minmax校准方法...")
    minmax_ranges = calibrate_model(model, calibration_data, method='minmax')
    
    # 创建新模型用于percentile校准
    model2 = create_dummy_model()
    
    print("使用percentile校准方法...")
    percentile_ranges = calibrate_model(model2, calibration_data, method='percentile', percentile=99.9)
    
    # 打印校准结果比较
    print("校准范围比较 (以第一个线性层为例):")
    if 'fc1' in minmax_ranges and 'fc1' in percentile_ranges:
        minmax_min, minmax_max = minmax_ranges['fc1']['min'], minmax_ranges['fc1']['max']
        percentile_min, percentile_max = percentile_ranges['fc1']['min'], percentile_ranges['fc1']['max']
        
        print(f"  minmax: min={minmax_min:.6f}, max={minmax_max:.6f}")
        print(f"  percentile: min={percentile_min:.6f}, max={percentile_max:.6f}")
        print(f"  范围差异: {abs((minmax_max - minmax_min) - (percentile_max - percentile_min)):.6f}")
    print()


def example_tensor_quantization():
    """
    张量量化示例
    """
    print("===== 张量量化示例 =====")
    
    # 创建一个随机张量
    tensor = torch.randn(1000)
    print(f"原始张量: 均值={tensor.mean():.4f}, 方差={tensor.var():.4f}")
    
    # 量化张量
    quantized_tensor, scale, zero_point = quantize_tensor(tensor, bits=8, symmetric=True)
    print(f"量化后: scale={scale:.6f}, zero_point={zero_point}")
    
    # 反量化张量
    dequantized_tensor = dequantize_tensor(quantized_tensor, scale, zero_point)
    print(f"反量化后: 均值={dequantized_tensor.mean():.4f}, 方差={dequantized_tensor.var():.4f}")
    
    # 计算量化误差
    mse = torch.nn.functional.mse_loss(dequantized_tensor, tensor).item()
    print(f"量化MSE误差: {mse:.6f}")
    print()


def example_int8_inference():
    """
    INT8推理示例
    """
    print("===== INT8推理示例 ====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 创建量化器并量化模型
    quantizer = PTQQuantizer(bits=8, symmetric=True)
    quantized_model, quant_params = quantizer.quantize_model(model, calibration_data)
    
    # 创建量化模型包装器
    int8_model = QuantizedModel(model, quant_params)
    
    # 测试输入
    test_input = torch.randn(1, 784)
    
    # 运行原始模型
    with torch.no_grad():
        original_output = model(test_input)
        
        # 运行INT8模型
        quantized_output = int8_model(test_input)
        
        # 模拟INT8推理
        simulated_output = simulate_int8_inference(model, test_input, quant_params)
    
    # 计算误差
    mse_vs_original = torch.nn.functional.mse_loss(quantized_output, original_output).item()
    mse_vs_simulated = torch.nn.functional.mse_loss(simulated_output, original_output).item()
    
    print(f"量化模型 vs 原始模型 MSE: {mse_vs_original:.6f}")
    print(f"模拟INT8 vs 原始模型 MSE: {mse_vs_simulated:.6f}")
    print()


def example_int8_matmul():
    """
    INT8矩阵乘法示例
    """
    print("===== INT8矩阵乘法示例 ====")
    
    # 创建输入和权重张量
    input_tensor = torch.randn(16, 128)
    weight_tensor = torch.randn(64, 128)
    
    # 量化输入和权重
    quant_input, input_scale, input_zero_point = quantize_tensor(input_tensor, bits=8)
    quant_weight, weight_scale, weight_zero_point = quantize_tensor(weight_tensor, bits=8)
    
    # 执行INT8矩阵乘法
    int8_result = run_int8_matmul(
        quant_input, quant_weight, 
        input_scale, weight_scale, 
        input_zero_point, weight_zero_point
    )
    
    # 执行浮点矩阵乘法作为参考
    float_result = torch.matmul(input_tensor, weight_tensor.t())
    
    # 计算误差
    mse = torch.nn.functional.mse_loss(int8_result, float_result).item()
    
    print(f"INT8矩阵乘法 vs 浮点矩阵乘法 MSE: {mse:.6f}")
    print(f"输入量化参数: scale={input_scale:.6f}, zero_point={input_zero_point}")
    print(f"权重量化参数: scale={weight_scale:.6f}, zero_point={weight_zero_point}")
    print()


def example_accuracy_comparison():
    """
    量化前后精度比较示例
    """
    print("===== 量化前后精度比较示例 ====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 量化模型
    quantized_model = quantize_model(model, calibration_data)
    
    # 准备测试数据
    test_data = generate_calibration_data(5)
    
    # 比较精度
    accuracy_metrics = compare_quantization_accuracy(model, quantized_model, test_data)
    
    print(f"平均MSE误差: {accuracy_metrics['average_mse']:.6f}")
    print(f"最大MSE误差: {accuracy_metrics['max_mse']:.6f}")
    print(f"最小MSE误差: {accuracy_metrics['min_mse']:.6f}")
    print()


def example_save_load_quantized_model():
    """
    保存和加载量化模型示例
    """
    print("===== 保存和加载量化模型示例 ====")
    
    # 创建模型和校准数据
    model = create_dummy_model()
    calibration_data = generate_calibration_data(10)
    
    # 量化模型
    quantized_model = quantize_model(model, calibration_data)
    
    # 保存量化模型
    model_path = "quantized_model.pt"
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quant_params': getattr(quantized_model, 'quant_params', {})
    }, model_path)
    
    print(f"量化模型已保存到 {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    # 加载量化模型
    loaded_data = torch.load(model_path)
    new_model = create_dummy_model()
    new_model.load_state_dict(loaded_data['model_state_dict'])
    new_model.quant_params = loaded_data['quant_params']
    
    # 验证加载的模型
    test_input = torch.randn(1, 784)
    with torch.no_grad():
        original_output = quantized_model(test_input)
        loaded_output = new_model(test_input)
    
    # 检查输出是否一致
    output_diff = torch.max(torch.abs(original_output - loaded_output)).item()
    print(f"加载的模型与原始量化模型输出差异: {output_diff:.6f}")
    
    # 清理
    if os.path.exists(model_path):
        os.remove(model_path)
    print()


def run_all_examples():
    """
    运行所有示例
    """
    print("开始运行所有PTQ工具示例...\n")
    
    example_basic_quantization()
    example_custom_quantization_params()
    example_different_calibration_methods()
    example_tensor_quantization()
    example_int8_inference()
    example_int8_matmul()
    example_accuracy_comparison()
    example_save_load_quantized_model()
    
    print("所有示例运行完成!")


if __name__ == "__main__":
    run_all_examples()