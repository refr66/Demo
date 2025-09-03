import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 实例化掩码多头注意力机制
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 实例化编码器-解码器交叉注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 实例化前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 实例化层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 实例化dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力子层
        # 先进行层归一化，然后进行掩码自注意力计算（Pre-LN架构）
        x_norm = self.norm1(x)
        attn_output, attn_weights1 = self.masked_self_attn(x_norm, x_norm, x_norm, tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # 残差连接
        
        # 交叉注意力子层
        # 先进行层归一化，然后进行交叉注意力计算（Pre-LN架构）
        x_norm = self.norm2(x)
        attn_output, attn_weights2 = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        attn_output = self.dropout2(attn_output)
        x = x + attn_output  # 残差连接
        
        # 前馈网络子层
        # 先进行层归一化，然后进行前馈网络计算（Pre-LN架构）
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        ff_output = self.dropout3(ff_output)
        x = x + ff_output  # 残差连接
        
        return x, attn_weights1, attn_weights2