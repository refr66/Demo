import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os
import time

class DataParallel:
    """
    数据并行实现，支持多GPU训练
    将数据分割到不同设备，每个设备维护完整模型副本
    """
    def __init__(self, model, device_ids=None, backend='gloo'):
        """
        初始化数据并行包装器
        
        参数:
            model: 要并行化的PyTorch模型
            device_ids: 使用的设备ID列表，如果为None则使用所有可用设备
            backend: 分布式通信后端
        """
        self.original_model = model
        self.device_ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))
        self.num_devices = len(self.device_ids)
        self.backend = backend
        
        # 模型复制到每个设备
        self.models = nn.ModuleList()
        for device_id in self.device_ids:
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            model_copy = copy_model(model, device)
            self.models.append(model_copy)
        
        # 初始化分布式环境（如果使用多进程）
        self.distributed = self.num_devices > 1 and torch.cuda.is_available()
        if self.distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend=self.backend, rank=0, world_size=self.num_devices)
    
    def forward(self, x):
        """
        前向传播，将输入数据分割并分发到各个设备
        
        参数:
            x: 输入张量
        
        返回:
            聚合后的输出张量
        """
        if self.num_devices == 1:
            # 单设备情况，直接前向传播
            return self.models[0](x.to(self.device_ids[0]))
        
        # 分割输入数据
        inputs = torch.chunk(x, self.num_devices, dim=0)
        outputs = []
        
        # 在每个设备上执行前向传播
        for i, model in enumerate(self.models):
            device = torch.device(f'cuda:{self.device_ids[i]}' if torch.cuda.is_available() else 'cpu')
            input_device = inputs[i].to(device)
            output = model(input_device)
            outputs.append(output)
        
        # 聚合输出
        return torch.cat(outputs, dim=0)
    
    def backward(self, loss):
        """
        反向传播，计算并聚合梯度
        
        参数:
            loss: 损失张量
        """
        if self.num_devices == 1:
            # 单设备情况，直接反向传播
            loss.backward()
            return
        
        # 分割损失并在每个设备上执行反向传播
        losses = torch.chunk(loss, self.num_devices, dim=0)
        for i, (model, device_loss) in enumerate(zip(self.models, losses)):
            device = torch.device(f'cuda:{self.device_ids[i]}' if torch.cuda.is_available() else 'cpu')
            device_loss = device_loss.to(device)
            device_loss.backward()
    
    def step(self, optimizer):
        """
        更新模型参数，确保所有设备上的参数同步
        
        参数:
            optimizer: 优化器
        """
        if self.num_devices == 1:
            # 单设备情况，直接更新
            optimizer.step()
            return
        
        # 聚合所有设备的梯度
        self._average_gradients()
        
        # 在主设备上更新参数
        with torch.no_grad():
            for param in self.models[0].parameters():
                optimizer.step()
                # 将更新后的参数广播到其他设备
                for model in self.models[1:]:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data.copy_(param.data)
    
    def _average_gradients(self):
        """
        聚合所有设备上的梯度
        """
        for param_idx, param in enumerate(self.models[0].parameters()):
            if param.grad is None:
                continue
            
            # 收集所有设备的梯度
            grads = [param.grad.clone()]
            for model in self.models[1:]:
                device_param = list(model.parameters())[param_idx]
                if device_param.grad is not None:
                    grads.append(device_param.grad.clone())
            
            # 计算平均梯度
            avg_grad = torch.mean(torch.stack(grads), dim=0)
            
            # 将平均梯度应用到所有设备
            param.grad = avg_grad
            for model in self.models[1:]:
                device_param = list(model.parameters())[param_idx]
                device_param.grad = avg_grad.clone()
    
    def zero_grad(self):
        """
        清零所有设备上的梯度
        """
        for model in self.models:
            model.zero_grad()
    
    def state_dict(self):
        """
        获取模型状态字典
        
        返回:
            主设备上的模型状态字典
        """
        return self.models[0].state_dict()
    
    def load_state_dict(self, state_dict):
        """
        加载模型状态字典到所有设备
        
        参数:
            state_dict: 模型状态字典
        """
        for model in self.models:
            model.load_state_dict(state_dict)


def copy_model(model, device):
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


def create_distributed_loader(dataset, batch_size, shuffle=True):
    """
    创建分布式数据加载器
    
    参数:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    返回:
        分布式数据加载器
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(shuffle and sampler is None), 
        sampler=sampler
    )


def train_parallel(model, train_loader, criterion, optimizer, num_epochs=1, device_ids=None):
    """
    使用数据并行进行训练的便捷函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device_ids: 使用的设备ID列表
    
    返回:
        训练后的模型
    """
    # 创建数据并行包装器
    dp_model = DataParallel(model, device_ids)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            # 清零梯度
            dp_model.zero_grad()
            
            # 前向传播
            outputs = dp_model.forward(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            dp_model.backward(loss)
            
            # 更新参数
            dp_model.step(optimizer)
            
            # 统计损失
            running_loss += loss.item()
            
            # 打印进度
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 打印每轮训练时间
        end_time = time.time()
        print(f'Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds')
    
    # 返回训练后的模型
    return dp_model.models[0]