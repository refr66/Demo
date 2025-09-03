import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算位置编码的分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 为batch维度添加一个维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 将位置编码注册为缓冲区，这样它就不会被视为模型参数，但会随模型一起保存
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 将位置编码添加到输入嵌入中
        # x的形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :].detach()
        return x