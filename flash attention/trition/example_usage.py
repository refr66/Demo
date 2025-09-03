import sys
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入我们实现的Triton Flash Attention
from flash_attention.trition.flash_attention_triton import TritonMultiHeadAttention

# 导入原始的MultiHeadAttention进行比较
from transformer_src.multi_head_attention import MultiHeadAttention

# 导入EncoderLayer以演示集成
from transformer_src.encoder_layer import EncoderLayer

# 重写EncoderLayer以使用TritonMultiHeadAttention
class TritonEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TritonEncoderLayer, self).__init__()
        
        # 使用Triton实现的多头注意力机制
        self.self_attn = TritonMultiHeadAttention(d_model, num_heads)
        
        # 实例化前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
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

# 性能测试函数
def benchmark_attention(d_model=512, num_heads=8, seq_len=100, batch_size=4, device='cuda'):
    """
    比较标准注意力和Triton实现的注意力的性能
    """
    # 创建输入张量
    q = torch.randn(batch_size, seq_len, d_model).to(device)
    k = torch.randn(batch_size, seq_len, d_model).to(device)
    v = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 创建标准注意力和Triton注意力
    standard_attn = MultiHeadAttention(d_model, num_heads).to(device)
    triton_attn = TritonMultiHeadAttention(d_model, num_heads).to(device)
    
    # 预热运行
    for _ in range(3):
        with torch.no_grad():
            standard_attn(q, k, v)
            triton_attn(q, k, v)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # 测量标准注意力的时间
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次运行取平均
            standard_output, standard_weights = standard_attn(q, k, v)
    torch.cuda.synchronize() if device == 'cuda' else None
    standard_time = (time.time() - start_time) / 100
    
    # 测量Triton注意力的时间
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次运行取平均
            triton_output, triton_weights = triton_attn(q, k, v)
    torch.cuda.synchronize() if device == 'cuda' else None
    triton_time = (time.time() - start_time) / 100
    
    # 计算加速比
    speedup = standard_time / triton_time
    
    print(f"Performance comparison for d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}:")
    print(f"  Standard Attention: {standard_time*1000:.2f} ms")
    print(f"  Triton Attention: {triton_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return standard_time, triton_time, speedup

def benchmark_encoder_layer(d_model=512, num_heads=8, d_ff=2048, seq_len=100, batch_size=4, device='cuda'):
    """
    比较标准编码器层和使用Triton注意力的编码器层的性能
    """
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 创建标准编码器层和Triton编码器层
    standard_layer = EncoderLayer(d_model, num_heads, d_ff).to(device)
    triton_layer = TritonEncoderLayer(d_model, num_heads, d_ff).to(device)
    
    # 预热运行
    for _ in range(3):
        with torch.no_grad():
            standard_layer(x)
            triton_layer(x)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # 测量标准编码器层的时间
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次运行取平均
            standard_output, standard_weights = standard_layer(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    standard_time = (time.time() - start_time) / 100
    
    # 测量Triton编码器层的时间
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次运行取平均
            triton_output, triton_weights = triton_layer(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    triton_time = (time.time() - start_time) / 100
    
    # 计算加速比
    speedup = standard_time / triton_time
    
    print(f"Encoder Layer performance comparison for d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}:")
    print(f"  Standard Layer: {standard_time*1000:.2f} ms")
    print(f"  Triton Layer: {triton_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return standard_time, triton_time, speedup

def run_benchmark_suite():
    """
    运行完整的性能测试套件，测试不同配置下的性能
    """
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on device: {device}")
    
    # 测试不同的序列长度
    seq_lens = [100, 200, 400, 800]
    results_seq_len = []
    
    for seq_len in seq_lens:
        print(f"\nTesting with sequence length: {seq_len}")
        std_time, tri_time, speedup = benchmark_attention(seq_len=seq_len, device=device)
        results_seq_len.append((seq_len, speedup))
    
    # 测试不同的头数
    num_heads_list = [4, 8, 16, 32]
    results_heads = []
    
    for num_heads in num_heads_list:
        # 确保d_model能被num_heads整除
        d_model = 512 if 512 % num_heads == 0 else 256
        print(f"\nTesting with number of heads: {num_heads}, d_model: {d_model}")
        std_time, tri_time, speedup = benchmark_attention(num_heads=num_heads, d_model=d_model, device=device)
        results_heads.append((num_heads, speedup))
    
    # 绘制结果图
    plot_results(results_seq_len, results_heads)
    
    # 测试编码器层性能
    print(f"\nTesting Encoder Layer performance")
    benchmark_encoder_layer(device=device)

def plot_results(results_seq_len, results_heads):
    """
    绘制性能测试结果图
    """
    # 创建一个新的图形
    plt.figure(figsize=(12, 5))
    
    # 绘制序列长度与加速比的关系
    plt.subplot(1, 2, 1)
    seq_lens, speedups = zip(*results_seq_len)
    plt.plot(seq_lens, speedups, 'o-', linewidth=2)
    plt.title('Speedup vs. Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup')
    plt.grid(True)
    
    # 绘制头数与加速比的关系
    plt.subplot(1, 2, 2)
    num_heads, speedups = zip(*results_heads)
    plt.plot(num_heads, speedups, 's-', linewidth=2)
    plt.title('Speedup vs. Number of Heads')
    plt.xlabel('Number of Heads')
    plt.ylabel('Speedup')
    plt.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig('flash_attention_benchmark.png')
    print("Benchmark plot saved as 'flash_attention_benchmark.png'")

# 如何在现有项目中替换注意力机制的指南
def integration_guide():
    """
    提供如何在现有Transformer项目中集成Triton Flash Attention的指南
    """
    print("\n=== Integration Guide: How to Use Triton Flash Attention in Your Project ===")
    print("\n1. First, install the required dependencies:")
    print("   pip install -r flash_attention/trition/requirements.txt")
    print("\n2. Import the TritonMultiHeadAttention class:")
    print("   from flash_attention.trition.flash_attention_triton import TritonMultiHeadAttention")
    print("\n3. Replace the standard MultiHeadAttention with TritonMultiHeadAttention in your model:")
    print("   # Before:")
    print("   from transformer_src.multi_head_attention import MultiHeadAttention")
    print("   self.self_attn = MultiHeadAttention(d_model, num_heads)")
    print("\n   # After:")
    print("   from flash_attention.trition.flash_attention_triton import TritonMultiHeadAttention")
    print("   self.self_attn = TritonMultiHeadAttention(d_model, num_heads)")
    print("\n4. For more advanced usage, you can modify the EncoderLayer or DecoderLayer classes")
    print("   as shown in this example to use the Triton implementation.")
    print("\n5. Note: The Triton implementation currently works best with GPU devices.")
    print("   Make sure you have CUDA available for optimal performance.")

if __name__ == "__main__":
    # 确保使用PyTorch的JIT编译优化
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    
    print("=== Triton Flash Attention Example ===")
    
    # 运行基本的性能测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 运行基准性能测试
    print("\nRunning basic attention benchmark...")
    benchmark_attention(device=device)
    
    # 运行编码器层性能测试
    print("\nRunning encoder layer benchmark...")
    benchmark_encoder_layer(device=device)
    
    # 提供集成指南
    integration_guide()
    
    # 提示用户可以运行完整的基准测试套件
    print("\nTo run a comprehensive benchmark suite with different configurations,")
    print("uncomment the following line in the code:")
    print("# run_benchmark_suite()")