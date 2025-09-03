import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        # 实例化编码器
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # 实例化解码器
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # 输出层，将解码器的输出映射到目标词汇表空间
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码源序列
        enc_output, enc_attn_weights = self.encoder(src, src_mask)
        
        # 解码目标序列
        dec_output, dec_attn_weights, cross_attn_weights = self.decoder(
            tgt, enc_output, src_mask, tgt_mask
        )
        
        # 输出层变换
        output = self.output_layer(dec_output)  # [batch_size, tgt_seq_len, tgt_vocab_size]
        
        return output, enc_attn_weights, dec_attn_weights, cross_attn_weights