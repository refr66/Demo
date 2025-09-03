import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        # 设置模型参数
        self.d_model = d_model
        self.d_ff = d_ff  # 前馈网络内部层的维度
        self.dropout = nn.Dropout(dropout)
        
        # 定义前馈网络的两个线性层
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # 定义激活函数 (使用GELU可以获得更好的性能)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # 第一层线性变换 + 激活函数
        x = self.fc1(x)  # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二层线性变换
        x = self.fc2(x)  # [batch_size, seq_len, d_model]
        
        return x