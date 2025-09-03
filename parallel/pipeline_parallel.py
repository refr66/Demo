import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import List, Tuple, Dict, Any, Optional, Union

class PipelineParallel:
    """
    流水线并行实现，将模型层分配到不同设备
    通过流水线方式执行前向和反向传播
    """
    def __init__(self, model_layers: List[nn.Module], device_ids: List[int], chunk_size: int = 1):
        """
        初始化流水线并行包装器
        
        参数:
            model_layers: 模型层列表，将被分配到不同设备
            device_ids: 设备ID列表，与模型层一一对应
            chunk_size: 流水线批处理块大小
        """
        self.model_layers = model_layers
        self.device_ids = device_ids
        self.chunk_size = chunk_size
        self.num_layers = len(model_layers)
        self.num_devices = len(device_ids)
        
        # 验证参数
        if self.num_layers != self.num_devices:
            raise ValueError(f"模型层数({self.num_layers})必须与设备数({self.num_devices})相等")
        
        # 将模型层移动到对应设备
        for i, (layer, device_id) in enumerate(zip(model_layers, device_ids)):
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            self.model_layers[i] = layer.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用流水线方式执行
        
        参数:
            x: 输入张量
        
        返回:
            最终输出张量
        """
        # 按chunk_size分割批次
        batch_size = x.size(0)
        chunks = torch.chunk(x, max(1, batch_size // self.chunk_size), dim=0)
        num_chunks = len(chunks)
        
        # 存储中间激活值，用于反向传播
        self.activations = []
        
        # 使用1F1B策略执行流水线
        outputs = []
        
        for chunk_idx in range(num_chunks):
            # 当前chunk
            chunk = chunks[chunk_idx]
            
            # 前向传播
            layer_outputs = []
            current_input = chunk.to(torch.device(f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu'))
            
            for i, (layer, device_id) in enumerate(zip(self.model_layers, self.device_ids)):
                # 确保输入在正确的设备上
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                if current_input.device != device:
                    current_input = current_input.to(device)
                
                # 执行前向计算
                with torch.no_grad():
                    current_output = layer(current_input)
                
                # 保存激活值用于反向传播
                if i < self.num_layers - 1:
                    layer_outputs.append(current_input.detach().requires_grad_(True))
                
                current_input = current_output
            
            # 保存当前chunk的激活值和输出
            self.activations.append(layer_outputs)
            outputs.append(current_input)
        
        # 聚合所有chunk的输出
        final_output = torch.cat(outputs, dim=0)
        return final_output
    
    def backward(self, grad_output: torch.Tensor) -> None:
        """
        反向传播，使用流水线方式执行
        
        参数:
            grad_output: 输出梯度张量
        """
        # 按chunk_size分割梯度
        batch_size = grad_output.size(0)
        chunks = torch.chunk(grad_output, max(1, batch_size // self.chunk_size), dim=0)
        num_chunks = len(chunks)
        
        # 存储每层的梯度
        layer_grads = [None] * self.num_layers
        
        # 从最后一层开始反向传播
        for chunk_idx in range(num_chunks - 1, -1, -1):
            # 当前chunk的梯度
            chunk_grad = chunks[chunk_idx]
            
            # 当前chunk的激活值
            chunk_activations = self.activations[chunk_idx]
            
            # 反向传播
            current_grad = chunk_grad.to(torch.device(f'cuda:{self.device_ids[-1]}' if torch.cuda.is_available() else 'cpu'))
            
            for i in range(self.num_layers - 1, -1, -1):
                layer = self.model_layers[i]
                device_id = self.device_ids[i]
                device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
                
                # 确保梯度在正确的设备上
                if current_grad.device != device:
                    current_grad = current_grad.to(device)
                
                if i == 0:
                    # 第一层没有输入激活值，直接反向传播
                    with torch.enable_grad():
                        output = layer(torch.zeros_like(chunk_activations[0], device=device))
                        output.backward(current_grad)
                elif i == self.num_layers - 1:
                    # 最后一层，使用输入激活值和输出梯度
                    with torch.enable_grad():
                        output = layer(chunk_activations[i-1])
                        output.backward(current_grad)
                        current_grad = chunk_activations[i-1].grad
                else:
                    # 中间层
                    with torch.enable_grad():
                        output = layer(chunk_activations[i-1])
                        output.backward(current_grad)
                        current_grad = chunk_activations[i-1].grad
    
    def step(self, optimizer: optim.Optimizer) -> None:
        """
        更新模型参数
        
        参数:
            optimizer: 优化器
        """
        optimizer.step()
    
    def zero_grad(self) -> None:
        """
        清零所有模型层的梯度
        """
        for layer in self.model_layers:
            layer.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取模型状态字典
        
        返回:
            模型状态字典
        """
        state_dict = {}
        for i, layer in enumerate(self.model_layers):
            state_dict[f'layer_{i}'] = layer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载模型状态字典
        
        参数:
            state_dict: 模型状态字典
        """
        for i, layer in enumerate(self.model_layers):
            if f'layer_{i}' in state_dict:
                layer.load_state_dict(state_dict[f'layer_{i}'])


def split_model(model: nn.Module, device_ids: List[int]) -> Tuple[List[nn.Module], List[int]]:
    """
    将模型分割成多个子模块，分配到不同设备
    
    参数:
        model: 要分割的完整模型
        device_ids: 设备ID列表
    
    返回:
        (分割后的模型层列表, 设备ID列表)
    """
    # 获取模型的所有子模块
    modules = list(model.modules())
    # 排除顶层模型本身
    modules = modules[1:] if len(modules) > 1 else modules
    
    # 计算每层分配的设备数量
    num_layers = len(modules)
    num_devices = len(device_ids)
    
    # 如果层数少于设备数，重复使用设备
    if num_layers < num_devices:
        device_ids = device_ids[:num_layers]
    # 如果层数多于设备数，将多层分配到一个设备
    elif num_layers > num_devices:
        layer_per_device = num_layers // num_devices
        remainder = num_layers % num_devices
        
        new_modules = []
        new_device_ids = []
        start_idx = 0
        
        for i in range(num_devices):
            # 计算当前设备的层数
            count = layer_per_device + (1 if i < remainder else 0)
            end_idx = start_idx + count
            
            # 创建包含多个层的新模块
            if count == 1:
                # 只有一层，直接使用
                new_modules.append(modules[start_idx])
            else:
                # 多层，创建Sequential
                seq_module = nn.Sequential(*modules[start_idx:end_idx])
                new_modules.append(seq_module)
            
            new_device_ids.append(device_ids[i])
            start_idx = end_idx
        
        modules = new_modules
        device_ids = new_device_ids
    
    return modules, device_ids


def create_pipeline_model(model: nn.Module, device_ids: List[int], chunk_size: int = 1) -> PipelineParallel:
    """
    创建流水线并行模型的便捷函数
    
    参数:
        model: 要并行化的完整模型
        device_ids: 设备ID列表
        chunk_size: 流水线批处理块大小
    
    返回:
        流水线并行模型
    """
    # 分割模型
    model_layers, device_ids = split_model(model, device_ids)
    
    # 创建流水线并行模型
    return PipelineParallel(model_layers, device_ids, chunk_size)


def train_pipeline(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, 
                   num_epochs: int = 1, device_ids: List[int] = None, 
                   chunk_size: int = 1) -> nn.Module:
    """
    使用流水线并行进行训练的便捷函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device_ids: 设备ID列表
        chunk_size: 流水线批处理块大小
    
    返回:
        训练后的模型
    """
    # 如果没有指定设备ID，使用所有可用设备
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) == 0:
            device_ids = [0]  # CPU
    
    # 创建流水线并行模型
    pipeline_model = create_pipeline_model(model, device_ids, chunk_size)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            # 清零梯度
            pipeline_model.zero_grad()
            
            # 前向传播
            outputs = pipeline_model.forward(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            pipeline_model.backward(loss)
            
            # 更新参数
            pipeline_model.step(optimizer)
            
            # 统计损失
            running_loss += loss.item()
            
            # 打印进度
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 打印每轮训练时间
        end_time = time.time()
        print(f'Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds')
    
    # 重建原始模型结构
    # 注意：这里简化处理，实际应用中可能需要更复杂的重建逻辑
    with torch.no_grad():
        model.load_state_dict(pipeline_model.state_dict())
    
    return model