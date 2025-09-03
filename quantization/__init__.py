# Post-Training Quantization (PTQ) 工具包

# 版本信息
__version__ = "0.1.0"

# 导出核心量化功能
from .quantization import (
    Quantizer,
    PTQQuantizer,
    quantize_model,
    quantize_tensor,
    dequantize_tensor
)

# 导出校准功能
from .calibration import (
    Calibrator,
    create_minmax_calibrator,
    create_percentile_calibrator,
    create_histogram_calibrator,
    calibrate_model,
    get_calibration_stats,
    set_calibration_stats
)

# 导出推理功能
from .inference import (
    QuantizedLinear,
    IntegerMatrixMultiplication,
    PseudoQuantizedModel,
    QuantizedModel,
    simulate_int8_inference,
    run_int8_matmul,
    compare_quantization_accuracy
)

# 导出示例函数（仅在直接运行时使用）
from .examples import run_all_examples

# 快捷访问函数
def get_quantizer(bits: int = 8, symmetric: bool = True) -> PTQQuantizer:
    """
    获取量化器实例
    
    Args:
        bits: 量化位宽
        symmetric: 是否使用对称量化
    
    Returns:
        PTQQuantizer实例
    """
    return PTQQuantizer(bits=bits, symmetric=symmetric)

def get_calibrator(method: str = 'minmax', **kwargs) -> Calibrator:
    """
    获取校准器实例
    
    Args:
        method: 校准方法
        **kwargs: 其他校准参数
    
    Returns:
        Calibrator实例
    """
    if method == 'minmax':
        return create_minmax_calibrator()
    elif method == 'percentile':
        percentile = kwargs.get('percentile', 99.9)
        return create_percentile_calibrator(percentile)
    elif method == 'histogram':
        num_bins = kwargs.get('num_bins', 2048)
        return create_histogram_calibrator(num_bins)
    else:
        raise ValueError(f"不支持的校准方法: {method}")

# 包描述
__doc__ = """
Post-Training Quantization (PTQ) 工具包

这个工具包提供了将训练好的深度学习模型（如Transformer）的FP32/FP16权重转换为INT8的功能，
实现模型压缩和加速。主要功能包括：

1. 校准 (Calibration): 向模型输入少量样本数据，记录每一层激活值的动态范围
2. 缩放因子计算: 根据动态范围计算从浮点数到INT8的映射关系
3. 权重量化: 将模型的权重从浮点数转换为INT8整数
4. INT8推理模拟: 实现简单的INT8矩阵乘法，展示伪量化和真量化的区别

使用方法示例:

```python
import torch
from quantization import quantize_model

# 加载训练好的模型
model = torch.load('path/to/model.pt')

# 准备校准数据
calibration_data = [torch.randn(32, 784) for _ in range(10)]

# 量化模型
quantized_model = quantize_model(model, calibration_data)

# 使用量化模型进行推理
input_data = torch.randn(1, 784)
output = quantized_model(input_data)
```
"""

# 公共API列表
__all__ = [
    # 量化核心功能
    'Quantizer',
    'PTQQuantizer',
    'quantize_model',
    'quantize_tensor',
    'dequantize_tensor',
    
    # 校准功能
    'Calibrator',
    'create_minmax_calibrator',
    'create_percentile_calibrator',
    'create_histogram_calibrator',
    'calibrate_model',
    'get_calibration_stats',
    'set_calibration_stats',
    
    # 推理功能
    'QuantizedLinear',
    'IntegerMatrixMultiplication',
    'PseudoQuantizedModel',
    'QuantizedModel',
    'simulate_int8_inference',
    'run_int8_matmul',
    'compare_quantization_accuracy',
    
    # 快捷函数
    'get_quantizer',
    'get_calibrator',
    
    # 示例
    'run_all_examples',
    
    # 版本
    '__version__'
]