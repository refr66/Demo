import torch
import torch.nn as nn
import math
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子（可选，有些实现会对嵌入进行缩放）
        self.d_model = d_model
    
    def forward(self, src, src_mask=None):
        # 词嵌入并添加位置编码
        # src的形状: [batch_size, seq_len]
        x = self.embedding(src)  # [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)  # 缩放词嵌入
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 逐层通过编码器层
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attn_weights.append(attn_weights)
        return x, all_attn_weights