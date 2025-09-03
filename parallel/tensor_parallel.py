import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import List, Tuple, Dict, Any, Optional
import torch.distributed as dist
import os

class TensorParallel:
    """
    张量并行实现，将模型参数张量分割到不同设备
    通过协同计算实现矩阵乘法等操作的并行化
    """
    def __init__(self, model: nn.Module, device_ids: List[int], split_dim: int = 0, backend: str = 'gloo'):
        """
        初始化张量并行包装器
        
        参数:
            model: 要并行化的PyTorch模型
            device_ids: 使用的设备ID列表
            split_dim: 张量分割维度，0表示按行分割，1表示按列分割
            backend: 分布式通信后端
        """
        self.original_model = model
        self.device_ids = device_ids
        self.split_dim = split_dim
        self.num_devices = len(device_ids)
        self.backend = backend
        
        # 初始化分布式环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group(backend=self.backend, rank=0, world_size=self.num_devices)
        
        # 分割模型参数并分配到各设备
        self.models = nn.ModuleList()
        self.param_splits = {}
        
        for device_id in device_ids:
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            model_copy = copy_model(model, device)
            self.models.append(model_copy)
        
        # 分割权重张量
        self._split_parameters()
    
    def _split_parameters(self):
        """
        分割模型参数到不同设备
        """
        # 获取所有需要分割的参数（通常是线性层的权重）
        for name, param in self.original_model.named_parameters():
            if param.ndim == 2:  # 仅处理二维张量（如线性层权重）
                # 按split_dim分割张量
                splits = torch.chunk(param, self.num_devices, dim=self.split_dim)
                
                # 保存分割信息
                self.param_splits[name] = splits
                
                # 将分割后的参数分配到对应设备
                for i, (model, split) in enumerate(zip(self.models, splits)):
                    device = torch.device(f'cuda:{self.device_ids[i]}' if torch.cuda.is_available() else 'cpu')
                    with torch.no_grad():
                        # 查找模型中的对应参数并更新
                        for model_name, model_param in model.named_parameters():
                            if model_name == name:
                                model_param.data.copy_(split.to(device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行并行计算
        
        参数:
            x: 输入张量
        
        返回:
            聚合后的输出张量
        """
        if self.num_devices == 1:
            # 单设备情况，直接前向传播
            return self.models[0](x.to(self.device_ids[0]))
        
        # 分割输入数据（如果按行分割权重）
        inputs = []
        if self.split_dim == 0:  # 权重按行分割，输入需要广播到所有设备
            for device_id in self.device_ids:
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                inputs.append(x.to(device))
        else:  # 权重按列分割，输入需要分割
            splits = torch.chunk(x, self.num_devices, dim=1)  # 按特征维度分割
            for i, device_id in enumerate(self.device_ids):
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                inputs.append(splits[i].to(device))
        
        # 在每个设备上执行前向传播
        outputs = []
        for i, (model, input_tensor) in enumerate(zip(self.models, inputs)):
            with torch.no_grad():
                output = model(input_tensor)
            outputs.append(output)
        
        # 聚合输出
        if self.split_dim == 0:  # 权重按行分割，输出需要连接
            # 将所有输出收集到主设备
            for i in range(1, len(outputs)):
                outputs[0] = torch.cat([outputs[0], outputs[i].to(outputs[0].device)], dim=1)
            return outputs[0]
        else:  # 权重按列分割，输出需要求和
            # 计算所有输出的平均值
            result = outputs[0].clone()
            for i in range(1, len(outputs)):
                result += outputs[i].to(result.device)
            return result / self.num_devices
    
    def backward(self, loss: torch.Tensor):
        """
        反向传播，计算并聚合梯度
        
        参数:
            loss: 损失张量
        """
        if self.num_devices == 1:
            # 单设备情况，直接反向传播
            loss.backward()
            return
        
        # 在每个设备上执行反向传播
        for model in self.models:
            # 复制损失到对应设备
            device = next(model.parameters()).device
            device_loss = loss.clone().to(device)
            device_loss.backward(retain_graph=True)
    
    def step(self, optimizer: optim.Optimizer):
        """
        更新模型参数
        
        参数:
            optimizer: 优化器
        """
        if self.num_devices == 1:
            # 单设备情况，直接更新
            optimizer.step()
            return
        
        # 聚合梯度
        self._all_reduce_gradients()
        
        # 在主设备上更新参数
        optimizer.step()
        
        # 将更新后的参数分割并广播到其他设备
        self._split_parameters()
    
    def _all_reduce_gradients(self):
        """
        使用all-reduce操作聚合所有设备上的梯度
        """
        for name, param in self.models[0].named_parameters():
            if param.grad is not None:
                # 收集所有设备的梯度
                grads = [param.grad]
                for model in self.models[1:]:
                    for model_name, model_param in model.named_parameters():
                        if model_name == name and model_param.grad is not None:
                            grads.append(model_param.grad.to(grads[0].device))
                            break
                
                # 计算平均梯度
                avg_grad = torch.mean(torch.stack(grads), dim=0)
                
                # 更新所有设备的梯度
                param.grad = avg_grad
                for i, model in enumerate(self.models[1:]):
                    for model_name, model_param in model.named_parameters():
                        if model_name == name and model_param.grad is not None:
                            model_param.grad = avg_grad.to(model_param.device)
                            break
    
    def zero_grad(self):
        """
        清零所有设备上的梯度
        """
        for model in self.models:
            model.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取完整的模型状态字典
        
        返回:
            完整的模型状态字典
        """
        state_dict = self.original_model.state_dict().copy()
        
        # 合并分割的参数
        for name, splits in self.param_splits.items():
            if name in state_dict:
                # 将所有分割的参数收集到主设备并合并
                merged = splits[0].clone()
                for split in splits[1:]:
                    merged = torch.cat([merged, split.to(merged.device)], dim=self.split_dim)
                state_dict[name] = merged
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        加载模型状态字典并分割到各设备
        
        参数:
            state_dict: 完整的模型状态字典
        """
        # 更新原始模型的状态字典
        self.original_model.load_state_dict(state_dict)
        
        # 重新分割参数
        self._split_parameters()


def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    """
    将模型复制到指定设备
    
    参数:
        model: 要复制的模型
        device: 目标设备
    
    返回:
        复制到目标设备的模型
    """
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(device)


def create_tensor_parallel_model(model: nn.Module, device_ids: List[int] = None, 
                                split_dim: int = 0, backend: str = 'gloo') -> TensorParallel:
    """
    创建张量并行模型的便捷函数
    
    参数:
        model: 要并行化的模型
        device_ids: 设备ID列表
        split_dim: 张量分割维度
        backend: 分布式通信后端
    
    返回:
        张量并行模型
    """
    # 如果没有指定设备ID，使用所有可用设备
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) == 0:
            device_ids = [0]  # CPU
    
    return TensorParallel(model, device_ids, split_dim, backend)


def train_tensor_parallel(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                          criterion: nn.Module, optimizer: optim.Optimizer, 
                          num_epochs: int = 1, device_ids: List[int] = None, 
                          split_dim: int = 0) -> nn.Module:
    """
    使用张量并行进行训练的便捷函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device_ids: 设备ID列表
        split_dim: 张量分割维度
    
    返回:
        训练后的模型
    """
    # 创建张量并行模型
    tp_model = create_tensor_parallel_model(model, device_ids, split_dim)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            # 清零梯度
            tp_model.zero_grad()
            
            # 前向传播
            outputs = tp_model.forward(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            tp_model.backward(loss)
            
            # 更新参数
            tp_model.step(optimizer)
            
            # 统计损失
            running_loss += loss.item()
            
            # 打印进度
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 打印每轮训练时间
        end_time = time.time()
        print(f'Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds')
    
    # 返回完整的训练后模型
    with torch.no_grad():
        model.load_state_dict(tp_model.state_dict())
    
    return model

class ParallelLinear(nn.Module):
    """
    并行线性层实现，支持张量并行的线性变换
    """
    def __init__(self, in_features: int, out_features: int, device_ids: List[int], 
                 bias: bool = True, split_dim: int = 0):
        """
        初始化并行线性层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            device_ids: 设备ID列表
            bias: 是否使用偏置
            split_dim: 张量分割维度
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device_ids = device_ids
        self.split_dim = split_dim
        self.num_devices = len(device_ids)
        
        # 计算每个设备上的特征维度
        if split_dim == 0:  # 按输出特征分割
            self.out_features_per_device = out_features // self.num_devices
        else:  # 按输入特征分割
            self.in_features_per_device = in_features // self.num_devices
        
        # 创建各个设备上的线性层
        self.linear_layers = nn.ModuleList()
        for device_id in device_ids:
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            
            if split_dim == 0:
                # 输出特征分割
                linear = nn.Linear(in_features, self.out_features_per_device, bias=bias, device=device)
            else:
                # 输入特征分割
                linear = nn.Linear(self.in_features_per_device, out_features, bias=bias, device=device)
            
            self.linear_layers.append(linear)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
        
        返回:
            输出张量
        """
        if self.num_devices == 1:
            # 单设备情况
            device = torch.device(f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
            return self.linear_layers[0](x.to(device))
        
        # 多设备并行计算
        outputs = []
        
        if self.split_dim == 0:  # 输出特征分割
            # 输入广播到所有设备
            for i, (linear, device_id) in enumerate(zip(self.linear_layers, self.device_ids)):
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                input_device = x.to(device)
                outputs.append(linear(input_device))
            
            # 连接输出
            result = outputs[0]
            for i in range(1, len(outputs)):
                result = torch.cat([result, outputs[i].to(result.device)], dim=1)
        else:  # 输入特征分割
            # 分割输入
            inputs = torch.chunk(x, self.num_devices, dim=1)
            
            for i, (linear, device_id, input_chunk) in enumerate(zip(self.linear_layers, self.device_ids, inputs)):
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                input_device = input_chunk.to(device)
                outputs.append(linear(input_device))
            
            # 求和输出
            result = outputs[0]
            for i in range(1, len(outputs)):
                result += outputs[i].to(result.device)
        
        return result