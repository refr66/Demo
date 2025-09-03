import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # 实例化多头注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 实例化前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 实例化层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 实例化dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        # 先进行层归一化，然后进行自注意力计算（Pre-LN架构）
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # 残差连接
        
        # 前馈网络子层
        # 先进行层归一化，然后进行前馈网络计算（Pre-LN架构）
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # 残差连接
        
        return x, attn_weights