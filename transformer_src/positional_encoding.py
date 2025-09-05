import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 预计算位置编码表并存储为缓冲区
        # 使用一个布尔值标记是否已经初始化，避免重复初始化
        self.d_model = d_model
        self.max_len = max_len
        self.initialized = False
    
    def forward(self, x):
        # 只在第一次调用时计算位置编码，并且仅计算所需长度
        if not self.initialized or self.pe.size(1) < x.size(1):
            # 计算需要的最大长度
            required_len = max(self.max_len, x.size(1))
            
            # 创建位置编码表
            pe = torch.zeros(1, required_len, self.d_model, device=x.device)
            position = torch.arange(0, required_len, dtype=torch.float, device=x.device).unsqueeze(1)
            
            # 预计算div_term，避免重复计算
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            
            # 偶数位置使用sin，奇数位置使用cos
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            
            # 保存位置编码
            self.register_buffer('pe', pe)
            self.initialized = True
        
        # 添加位置编码到输入嵌入中（不需要detach，因为它是缓冲区而非参数）
        return x + self.pe[:, :x.size(1), :]

# 创建一个高效的位置编码函数，用于静态计算
def get_positional_encoding(seq_len, d_model, device='cpu'):
    """高效的位置编码计算函数"""
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(1, seq_len, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    
    return pe