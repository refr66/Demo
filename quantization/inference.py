import torch
import numpy as np
from typing import Dict, Tuple, List, Union

class QuantizedLinear(torch.nn.Module):
    """
    量化的线性层，支持INT8权重和浮点激活或INT8激活
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        初始化量化线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 使用int8存储权重
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8))
        
        # 浮点偏置
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化参数
        self.weight_scale = torch.nn.Parameter(torch.empty(1))
        self.weight_zero_point = torch.nn.Parameter(torch.empty(1, dtype=torch.int8))
        
        # 激活量化参数
        self.activation_scale = None
        self.activation_zero_point = None
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """初始化参数"""
        # 注意：在实际使用中，这些参数会被量化过程覆盖
        torch.nn.init.kaiming_uniform_(self.weight.to(torch.float32), a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.to(torch.float32))
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
        self.weight_scale.data.fill_(1.0)
        self.weight_zero_point.data.fill_(0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 输入张量
        
        Returns:
            输出张量
        """
        # 如果输入是INT8，需要先反量化
        if input.dtype == torch.int8 and self.activation_scale is not None:
            # 反量化输入
            input = (input.to(torch.float32) - self.activation_zero_point) * self.activation_scale
        
        # 反量化权重
        weight_dequantized = (self.weight.to(torch.float32) - self.weight_zero_point) * self.weight_scale
        
        # 执行浮点矩阵乘法
        output = torch.nn.functional.linear(input, weight_dequantized, self.bias)
        
        return output


class IntegerMatrixMultiplication:
    """
    整数矩阵乘法实现，用于模拟真量化推理
    """
    @staticmethod
    def matmul_int8(input: np.ndarray, weight: np.ndarray, 
                   input_scale: float, weight_scale: float, 
                   input_zero_point: int = 0, weight_zero_point: int = 0) -> np.ndarray:
        """
        INT8矩阵乘法
        
        Args:
            input: 输入矩阵 (INT8)
            weight: 权重矩阵 (INT8)
            input_scale: 输入的缩放因子
            weight_scale: 权重的缩放因子
            input_zero_point: 输入的零点偏移
            weight_zero_point: 权重的零点偏移
        
        Returns:
            结果矩阵 (浮点)
        """
        # 确保输入类型正确
        assert input.dtype == np.int8, "输入必须是INT8类型"
        assert weight.dtype == np.int8, "权重必须是INT8类型"
        
        # 计算真实缩放因子
        output_scale = input_scale * weight_scale
        
        # 计算偏移量贡献
        # (input - input_zero_point) @ (weight - weight_zero_point) = 
        # input @ weight - input_zero_point * sum(weight) - weight_zero_point * sum(input) + input_zero_point * weight_zero_point * input.shape[1]
        
        # 计算整数矩阵乘法
        int_result = np.matmul(input.astype(np.int32), weight.astype(np.int32).T)
        
        # 减去零点偏移量的贡献
        sum_weights = np.sum(weight, axis=1, dtype=np.int32)
        sum_inputs = np.sum(input, axis=1, dtype=np.int32).reshape(-1, 1)
        
        int_result -= input_zero_point * sum_weights
        int_result -= weight_zero_point * sum_inputs
        int_result += input_zero_point * weight_zero_point * input.shape[1]
        
        # 应用输出缩放
        result = int_result.astype(np.float32) * output_scale
        
        return result
    
    @staticmethod
    def torch_matmul_int8(input: torch.Tensor, weight: torch.Tensor, 
                         input_scale: float, weight_scale: float, 
                         input_zero_point: int = 0, weight_zero_point: int = 0) -> torch.Tensor:
        """
        PyTorch版INT8矩阵乘法
        
        Args:
            input: 输入张量 (INT8)
            weight: 权重张量 (INT8)
            input_scale: 输入的缩放因子
            weight_scale: 权重的缩放因子
            input_zero_point: 输入的零点偏移
            weight_zero_point: 权重的零点偏移
        
        Returns:
            结果张量 (浮点)
        """
        # 确保输入类型正确
        assert input.dtype == torch.int8, "输入必须是INT8类型"
        assert weight.dtype == torch.int8, "权重必须是INT8类型"
        
        # 计算真实缩放因子
        output_scale = input_scale * weight_scale
        
        # 转换为int32进行计算
        input_int32 = input.to(torch.int32)
        weight_int32 = weight.to(torch.int32)
        
        # 计算整数矩阵乘法
        int_result = torch.matmul(input_int32, weight_int32.t())
        
        # 减去零点偏移量的贡献
        sum_weights = torch.sum(weight_int32, dim=1)
        sum_inputs = torch.sum(input_int32, dim=1).view(-1, 1)
        
        int_result -= input_zero_point * sum_weights
        int_result -= weight_zero_point * sum_inputs
        int_result += input_zero_point * weight_zero_point * input.shape[1]
        
        # 应用输出缩放
        result = int_result.to(torch.float32) * output_scale
        
        return result


class PseudoQuantizedModel:
    """
    伪量化模型包装器，在浮点计算中模拟量化效果
    """
    def __init__(self, model: torch.nn.Module, quant_params: Dict):
        """
        初始化伪量化模型
        
        Args:
            model: 原始PyTorch模型
            quant_params: 量化参数
        """
        self.model = model
        self.quant_params = quant_params
        
        # 注册前向钩子实现伪量化
        self._register_forward_hooks()
    
    def _register_forward_hooks(self) -> None:
        """注册前向钩子实现伪量化"""
        self.hooks = []
        
        def hook_fn(module_name):
            def hook(module, input, output):
                if module_name in self.quant_params and hasattr(module, 'weight'):
                    # 对权重应用伪量化
                    weight = module.weight.data
                    scale = self.quant_params[module_name]['scale']
                    zero_point = self.quant_params[module_name]['zero_point']
                    
                    # 模拟量化
                    quantized_weight = torch.clamp(torch.round(weight / scale + zero_point), -128, 127)
                    # 反量化
                    dequantized_weight = (quantized_weight - zero_point) * scale
                    
                    # 使用反量化后的权重
                    module.weight.data = dequantized_weight
                    
                # 对激活值应用伪量化（可选）
                # 注意：这会影响模型的精度，通常只在QAT中使用
                
            return hook
        
        # 为每个模块注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def __call__(self, *args, **kwargs):
        """调用原始模型"""
        return self.model(*args, **kwargs)
    
    def __del__(self):
        """移除钩子"""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()


class QuantizedModel:
    """
    量化模型包装器，使用INT8权重和浮点激活
    """
    def __init__(self, model: torch.nn.Module, quant_params: Dict):
        """
        初始化量化模型
        
        Args:
            model: 原始PyTorch模型
            quant_params: 量化参数
        """
        self.model = model
        self.quant_params = quant_params
        self._convert_to_quantized_layers()
    
    def _convert_to_quantized_layers(self) -> None:
        """将模型的线性层转换为量化线性层"""
        # 注意：这是一个简化的实现，实际中可能需要更复杂的模型转换逻辑
        for name, module in list(self.model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name in self.quant_params:
                # 创建量化线性层
                quant_linear = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None
                )
                
                # 复制量化参数
                quant_linear.weight.data = module.weight.data.to(torch.int8)
                if module.bias is not None:
                    quant_linear.bias.data = module.bias.data
                
                quant_linear.weight_scale.data = torch.tensor(self.quant_params[name]['scale'])
                quant_linear.weight_zero_point.data = torch.tensor(self.quant_params[name]['zero_point'], dtype=torch.int8)
                
                # 替换原始层
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model
                
                setattr(parent, child_name, quant_linear)
    
    def __call__(self, *args, **kwargs):
        """调用量化模型"""
        return self.model(*args, **kwargs)


# 便捷函数
def simulate_int8_inference(model: torch.nn.Module, input_data: torch.Tensor, 
                           quant_params: Dict) -> torch.Tensor:
    """
    模拟INT8推理
    
    Args:
        model: 原始PyTorch模型
        input_data: 输入数据
        quant_params: 量化参数
    
    Returns:
        推理结果
    """
    # 创建伪量化模型
    pseudo_quant_model = PseudoQuantizedModel(model, quant_params)
    
    # 运行推理
    with torch.no_grad():
        output = pseudo_quant_model(input_data)
    
    return output


def run_int8_matmul(input: torch.Tensor, weight: torch.Tensor, 
                    input_scale: float, weight_scale: float, 
                    input_zero_point: int = 0, weight_zero_point: int = 0) -> torch.Tensor:
    """
    运行INT8矩阵乘法
    
    Args:
        input: 输入张量
        weight: 权重张量
        input_scale: 输入的缩放因子
        weight_scale: 权重的缩放因子
        input_zero_point: 输入的零点偏移
        weight_zero_point: 权重的零点偏移
    
    Returns:
        矩阵乘法结果
    """
    return IntegerMatrixMultiplication.torch_matmul_int8(
        input, weight, input_scale, weight_scale, input_zero_point, weight_zero_point
    )


def compare_quantization_accuracy(original_model: torch.nn.Module, 
                                  quantized_model: torch.nn.Module, 
                                  test_data: List[torch.Tensor]) -> Dict[str, float]:
    """
    比较量化前后的模型精度
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        test_data: 测试数据
    
    Returns:
        精度指标字典
    """
    original_model.eval()
    quantized_model.eval()
    
    mse_values = []
    
    with torch.no_grad():
        for data in test_data:
            # 获取原始模型输出
            if isinstance(data, (list, tuple)):
                original_output = original_model(*data)
                quantized_output = quantized_model(*data)
            else:
                original_output = original_model(data)
                quantized_output = quantized_model(data)
            
            # 计算MSE
            mse = torch.nn.functional.mse_loss(quantized_output, original_output).item()
            mse_values.append(mse)
    
    # 计算平均MSE和相对误差
    avg_mse = np.mean(mse_values)
    
    return {
        'average_mse': avg_mse,
        'max_mse': np.max(mse_values),
        'min_mse': np.min(mse_values)
    }