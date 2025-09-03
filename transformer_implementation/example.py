import torch
import torch.nn as nn
from transformer import Transformer
from utils import create_src_mask, create_tgt_mask, plot_attention_weights

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 定义模型参数
src_vocab_size = 10000  # 源语言词汇表大小
tgt_vocab_size = 10000  # 目标语言词汇表大小
d_model = 512          # 模型维度
num_heads = 8          # 多头注意力的头数
num_layers = 6         # 编码器和解码器的层数
d_ff = 2048            # 前馈网络内部层的维度
dropout = 0.1          # Dropout概率
max_len = 100          # 最大序列长度

# 创建Transformer模型
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len
)

# 随机生成示例输入数据
batch_size = 2
src_seq_len = 10
tgt_seq_len = 12

# 生成随机的源序列和目标序列
src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))  # 避免使用0（假设0是填充标记）
tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))  # 避免使用0

# 定义填充标记的索引
pad_idx = 0

print(f"DEBUG: The shape of 'src' tensor is: {src.shape}") # <--- 请务必添加这一行！
# 创建掩码
src_mask = create_src_mask(src, pad_idx)
tgt_mask = create_tgt_mask(tgt, pad_idx)

# 前向传播
output, enc_attn_weights, dec_attn_weights, cross_attn_weights = model(src, tgt, src_mask, tgt_mask)

# 输出结果形状
print(f"Output shape: {output.shape}")

# 打印模型结构
print("\nTransformer Model Structure:")
print(model)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal number of trainable parameters: {total_params:,}")

# 简单的词汇表（用于演示目的）
class SimpleVocab:
    def __init__(self):
        self.itos = {i: f'word_{i}' for i in range(100)}

# 创建简单词汇表实例
vocab = SimpleVocab()

# 仅用于演示：打印第一个样本的部分输出
if output.size(0) > 0:
    # 获取第一个样本的输出
    first_output = output[0]
    # 获取预测的token索引（取最大值的索引）
    predicted_tokens = torch.argmax(first_output, dim=-1)
    
    print("\nSample input and predicted output:")
    print(f"Source sequence (indices): {src[0][:5].tolist()} ...")
    print(f"Target sequence (indices): {tgt[0][:5].tolist()} ...")
    print(f"Predicted sequence (indices): {predicted_tokens[:5].tolist()} ...")

# 提示用户如何可视化注意力权重
print("\nTo visualize attention weights, you can use the plot_attention_weights function.")
print("For example:")
print("plot_attention_weights(cross_attn_weights, src[0], tgt[0], layer=0, head=0, src_vocab=vocab, tgt_vocab=vocab)")

# 提示用户如何扩展该示例
print("\nTo use this model for actual tasks:")
print("1. Replace the random data with your actual dataset")
print("2. Add a training loop with optimizer and loss function")
print("3. Implement data preprocessing and tokenization")
print("4. Use the model for inference after training")