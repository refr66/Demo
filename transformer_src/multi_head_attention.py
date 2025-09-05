import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # 确保d_model能被num_heads整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 设置模型参数
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将掩码位置的分数设为极小值
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 加权求和得到输出
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len_q, d_k]
        
        return output, attention_weights
    
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 获取序列长度
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        # 线性变换并分割为多头
        Q = self.W_q(q).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, d_k]
        K = self.W_k(k).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, d_k]
        V = self.W_v(v).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, d_k]
        
        # 应用掩码（如果提供）
        # if mask is not None:
            # 调整掩码形状以匹配多头注意力的输入形状
            # mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
        
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 连接多头并通过输出层
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.W_o(attn_output)  # [batch_size, seq_len_q, d_model]
        
        return output, attn_weights