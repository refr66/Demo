import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

class Quantizer:
    """
    基础量化器类，提供量化和反量化功能
    """
    def __init__(self, bits: int = 8, symmetric: bool = True):
        """
        初始化量化器
        
        Args:
            bits: 量化位宽
            symmetric: 是否使用对称量化
        """
        self.bits = bits
        self.symmetric = symmetric
        
        # 计算量化范围
        if symmetric:
            self.min_val = -2 ** (bits - 1)
            self.max_val = 2 ** (bits - 1) - 1
        else:
            self.min_val = 0
            self.max_val = 2 ** bits - 1
        
    def calculate_scale(self, min_val: float, max_val: float) -> Tuple[float, float]:
        """
        计算缩放因子和零点
        
        Args:
            min_val: 原始数据最小值
            max_val: 原始数据最大值
        
        Returns:
            scale: 缩放因子
            zero_point: 零点偏移量
        """
        # 避免除零错误
        if max_val - min_val < 1e-8:
            max_val = min_val + 1e-8
            
        if self.symmetric:
            # 对称量化，零点固定为0
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / self.max_val
            zero_point = 0
        else:
            # 非对称量化
            scale = (max_val - min_val) / (self.max_val - self.min_val)
            zero_point = self.min_val - min_val / scale
            # 确保零点在量化范围内
            zero_point = max(self.min_val, min(self.max_val, zero_point))
            zero_point = round(zero_point)
            
        return scale, zero_point
    
    def quantize(self, tensor: Union[torch.Tensor, np.ndarray], scale: float, 
                 zero_point: float) -> Union[torch.Tensor, np.ndarray]:
        """
        将浮点张量量化为整数
        
        Args:
            tensor: 浮点张量
            scale: 缩放因子
            zero_point: 零点偏移量
        
        Returns:
            量化后的整数张量
        """
        # 应用缩放和偏移
        quantized = tensor / scale + zero_point
        # 截断到量化范围
        quantized = np.clip(quantized, self.min_val, self.max_val)
        # 四舍五入到整数
        quantized = np.round(quantized)
        
        if isinstance(tensor, torch.Tensor):
            return torch.tensor(quantized, dtype=torch.int8)
        else:
            return quantized.astype(np.int8)
    
    def dequantize(self, quantized: Union[torch.Tensor, np.ndarray], scale: float, 
                   zero_point: float) -> Union[torch.Tensor, np.ndarray]:
        """
        将量化的整数张量反量化为浮点数
        
        Args:
            quantized: 量化的整数张量
            scale: 缩放因子
            zero_point: 零点偏移量
        
        Returns:
            反量化后的浮点张量
        """
        # 应用反量化公式
        if isinstance(quantized, torch.Tensor):
            return (quantized.type(torch.float32) - zero_point) * scale
        else:
            return (quantized.astype(np.float32) - zero_point) * scale


class PTQQuantizer:
    """
    Post-Training Quantization (PTQ) 量化器
    """
    def __init__(self, bits: int = 8, symmetric: bool = True, 
                 calibration_method: str = 'minmax'):
        """
        初始化PTQ量化器
        
        Args:
            bits: 量化位宽
            symmetric: 是否使用对称量化
            calibration_method: 校准方法，支持'minmax'或'percentile'
        """
        self.quantizer = Quantizer(bits=bits, symmetric=symmetric)
        self.calibration_method = calibration_method
        self.stats = {}
        
    def collect_stats(self, model: torch.nn.Module, 
                     calibration_data: List[torch.Tensor]) -> None:
        """
        收集模型各层的统计信息
        
        Args:
            model: PyTorch模型
            calibration_data: 校准数据列表
        """
        # 注册前向钩子收集激活值统计信息
        hooks = []
        
        def hook_fn(module_name):
            def hook(module, input, output):
                if module_name not in self.stats:
                    self.stats[module_name] = {}
                
                # 收集权重统计信息
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data.cpu().numpy()
                    self.stats[module_name]['weight_min'] = weight.min()
                    self.stats[module_name]['weight_max'] = weight.max()
                    
                # 收集激活值统计信息
                if isinstance(output, torch.Tensor):
                    output = [output]
                
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        out_data = out.data.cpu().numpy()
                        if f'activation_{i}_min' not in self.stats[module_name]:
                            self.stats[module_name][f'activation_{i}_min'] = out_data.min()
                            self.stats[module_name][f'activation_{i}_max'] = out_data.max()
                        else:
                            self.stats[module_name][f'activation_{i}_min'] = min(
                                self.stats[module_name][f'activation_{i}_min'], out_data.min())
                            self.stats[module_name][f'activation_{i}_max'] = max(
                                self.stats[module_name][f'activation_{i}_max'], out_data.max())
            return hook
        
        # 为每个模块注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, 
                                  torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 运行校准数据
        model.eval()
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, (list, tuple)):
                    model(*data)
                else:
                    model(data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
    
    def quantize_weights(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """
        量化模型的权重
        
        Args:
            model: PyTorch模型
        
        Returns:
            量化参数字典
        """
        quant_params = {}
        
        for name, module in model.named_modules():
            if name in self.stats and hasattr(module, 'weight') and module.weight is not None:
                # 获取权重统计信息
                min_val = self.stats[name]['weight_min']
                max_val = self.stats[name]['weight_max']
                
                # 计算量化参数
                scale, zero_point = self.quantizer.calculate_scale(min_val, max_val)
                
                # 量化权重
                weight = module.weight.data.cpu().numpy()
                quantized_weight = self.quantizer.quantize(weight, scale, zero_point)
                
                # 保存量化参数
                quant_params[name] = {
                    'scale': scale,
                    'zero_point': zero_point,
                    'weight_shape': weight.shape
                }
                
                # 替换原始权重为量化权重
                # 注意：这里使用torch.int8类型，但实际存储的是量化后的值
                module.weight.data = torch.tensor(quantized_weight, dtype=torch.int8, 
                                                 device=module.weight.device)
        
        return quant_params
    
    def quantize_model(self, model: torch.nn.Module, 
                      calibration_data: List[torch.Tensor]) -> Tuple[torch.nn.Module, Dict]:
        """
        量化整个模型
        
        Args:
            model: PyTorch模型
            calibration_data: 校准数据列表
        
        Returns:
            量化后的模型
            量化参数字典
        """
        # 收集统计信息
        self.collect_stats(model, calibration_data)
        
        # 量化权重
        quant_params = self.quantize_weights(model)
        
        # 保存量化参数到模型属性
        model.quant_params = quant_params
        
        return model, quant_params


# 便捷函数
def quantize_model(model: torch.nn.Module, calibration_data: List[torch.Tensor], 
                  bits: int = 8, symmetric: bool = True) -> torch.nn.Module:
    """
    量化模型的便捷函数
    
    Args:
        model: PyTorch模型
        calibration_data: 校准数据列表
        bits: 量化位宽
        symmetric: 是否使用对称量化
    
    Returns:
        量化后的模型
    """
    quantizer = PTQQuantizer(bits=bits, symmetric=symmetric)
    quantized_model, _ = quantizer.quantize_model(model, calibration_data)
    return quantized_model


def quantize_tensor(tensor: torch.Tensor, bits: int = 8, 
                    symmetric: bool = True) -> Tuple[torch.Tensor, float, float]:
    """
    量化单个张量
    
    Args:
        tensor: 要量化的张量
        bits: 量化位宽
        symmetric: 是否使用对称量化
    
    Returns:
        量化后的张量
        缩放因子
        零点偏移量
    """
    quantizer = Quantizer(bits=bits, symmetric=symmetric)
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    scale, zero_point = quantizer.calculate_scale(min_val, max_val)
    quantized = quantizer.quantize(tensor, scale, zero_point)
    return quantized, scale, zero_point


def dequantize_tensor(quantized: torch.Tensor, scale: float, 
                      zero_point: float) -> torch.Tensor:
    """
    反量化单个张量
    
    Args:
        quantized: 量化的张量
        scale: 缩放因子
        zero_point: 零点偏移量
    
    Returns:
        反量化后的张量
    """
    quantizer = Quantizer()  # 使用默认参数
    return quantizer.dequantize(quantized, scale, zero_point)