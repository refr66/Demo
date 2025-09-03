import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# 导入CUDA版本的Flash Attention
from flash_attention_cuda import FlashAttention, CUDAMultiHeadAttention

# 导入标准的MultiHeadAttention用于比较
from transformer_src.multi_head_attention import MultiHeadAttention

# 定义自定义EncoderLayer，使用CUDA版本的注意力机制
class CUDAEncoderLayer(nn.Module):
    """
    自定义EncoderLayer，使用CUDA版本的Flash Attention
    用于性能比较
    """
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CUDAEncoderLayer, self).__init__()
        # 使用CUDA版本的注意力机制
        self.self_attn = CUDAMultiHeadAttention(d_model, num_heads, dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 激活函数
        self.activation = nn.GELU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src

# 定义标准的EncoderLayer，使用PyTorch标准注意力机制
class StandardEncoderLayer(nn.Module):
    """
    标准EncoderLayer，使用PyTorch标准的MultiHeadAttention
    用于性能比较
    """
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(StandardEncoderLayer, self).__init__()
        # 使用标准的注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 激活函数
        self.activation = nn.GELU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src

# 性能测试函数
def benchmark_attention(device, d_model=512, num_heads=8, seq_len=100):
    """
    比较标准注意力机制和CUDA优化版本的性能
    """
    # 创建测试数据
    batch_size = 2
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 初始化标准注意力机制
    standard_attn = MultiHeadAttention(d_model, num_heads).to(device)
    
    # 初始化CUDA优化的注意力机制
    cuda_attn = CUDAMultiHeadAttention(d_model, num_heads).to(device)
    
    # 预热
    for _ in range(5):
        with torch.no_grad():
            standard_attn(q, k, v)
            cuda_attn(q, k, v)
    
    # 确保CUDA同步
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 测试标准注意力
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            standard_attn(q, k, v)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    standard_time = (time.time() - start_time) / 100
    
    # 测试CUDA优化的注意力
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            cuda_attn(q, k, v)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    cuda_time = (time.time() - start_time) / 100
    
    # 计算加速比
    speedup = standard_time / cuda_time
    
    print(f"Performance comparison for d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}:")
    print(f"  Standard Attention: {standard_time*1000:.2f} ms")
    print(f"  CUDA Attention: {cuda_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return standard_time, cuda_time, speedup

# 编码器层性能测试
def benchmark_encoder_layer(device, d_model=512, num_heads=8, seq_len=100):
    """
    比较标准编码器层和使用CUDA优化注意力的编码器层的性能
    """
    # 创建测试数据
    batch_size = 2
    src = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 初始化标准编码器层
    standard_layer = StandardEncoderLayer(d_model, num_heads).to(device)
    
    # 初始化CUDA优化的编码器层
    cuda_layer = CUDAEncoderLayer(d_model, num_heads).to(device)
    
    # 预热
    for _ in range(5):
        with torch.no_grad():
            standard_layer(src)
            cuda_layer(src)
    
    # 确保CUDA同步
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 测试标准编码器层
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            standard_layer(src)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    standard_time = (time.time() - start_time) / 100
    
    # 测试CUDA优化的编码器层
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            cuda_layer(src)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    cuda_time = (time.time() - start_time) / 100
    
    # 计算加速比
    speedup = standard_time / cuda_time
    
    print(f"Encoder Layer performance comparison for d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}:")
    print(f"  Standard Layer: {standard_time*1000:.2f} ms")
    print(f"  CUDA Layer: {cuda_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return standard_time, cuda_time, speedup

# 不同序列长度的性能测试
def benchmark_sequence_lengths(device):
    """
    在不同序列长度下测试性能
    """
    d_model = 512
    num_heads = 8
    seq_lengths = [10, 50, 100, 200, 500, 1000]
    
    standard_times = []
    cuda_times = []
    
    print(f"\nBenchmarking different sequence lengths with d_model={d_model}, num_heads={num_heads}:")
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        std_time, cuda_time, _ = benchmark_attention(device, d_model, num_heads, seq_len)
        standard_times.append(std_time * 1000)  # 转换为毫秒
        cuda_times.append(cuda_time * 1000)    # 转换为毫秒
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, standard_times, marker='o', label='Standard Attention')
    plt.plot(seq_lengths, cuda_times, marker='s', label='CUDA Flash Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Performance vs. Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_vs_seq_len.png')
    print("\nPerformance plot saved as 'performance_vs_seq_len.png'")

# 完整的基准测试套件
def run_benchmark_suite(device):
    """
    运行完整的基准测试套件
    """
    print("\n=== Running Complete Benchmark Suite ===")
    
    # 测试不同的模型维度
    for d_model in [256, 512, 1024]:
        benchmark_attention(device, d_model=d_model)
        print()
    
    # 测试不同的头数
    for num_heads in [4, 8, 16]:
        benchmark_attention(device, num_heads=num_heads)
        print()
    
    # 测试不同的序列长度
    benchmark_sequence_lengths(device)

# 如何在现有项目中替换注意力机制的指南
def integration_guide():
    """
    提供如何在现有Transformer项目中集成CUDA Flash Attention的指南
    """
    print("\n=== Integration Guide: How to Use CUDA Flash Attention in Your Project ===")
    print("\n1. First, install the required dependencies:")
    print("   pip install -r flash_attention/cuda/requirements.txt")
    print("\n2. Compile the CUDA extension:")
    print("   cd flash_attention/cuda")
    print("   python setup.py install")
    print("\n3. Import the CUDAMultiHeadAttention class:")
    print("   from flash_attention.cuda.flash_attention_cuda import CUDAMultiHeadAttention")
    print("\n4. Replace the standard MultiHeadAttention with CUDAMultiHeadAttention in your model:")
    print("   # Before:")
    print("   from transformer_src.multi_head_attention import MultiHeadAttention")
    print("   self.self_attn = MultiHeadAttention(d_model, num_heads)")
    print("\n   # After:")
    print("   from flash_attention.cuda.flash_attention_cuda import CUDAMultiHeadAttention")
    print("   self.self_attn = CUDAMultiHeadAttention(d_model, num_heads)")
    print("\n5. For more advanced usage, you can modify the EncoderLayer or DecoderLayer classes")
    print("   as shown in this example to use the CUDA implementation.")
    print("\n6. Note: The CUDA implementation requires a CUDA-capable GPU for optimal performance.")

if __name__ == "__main__":
    # 确保使用PyTorch的JIT编译优化
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    
    print("=== CUDA Flash Attention Example ===")
    
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
    print("# run_benchmark_suite(device)")