import torch
import torch.nn as nn
import math

# 尝试导入CUDA扩展
CUDA_EXTENSION_AVAILABLE = False
try:
    import flash_attention_cuda_ext
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    print("CUDA extension not found. Please compile it first.")
    
# 如果CUDA扩展不可用，提供一个基于PyTorch的回退实现
def flash_attention_fallback(q, k, v, sm_scale=None):
    """
    基于PyTorch的Flash Attention回退实现
    当CUDA扩展不可用时使用
    """
    # 获取输入张量的维度
    batch_size, num_heads, seq_len_q, dim_per_head = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # 设置缩放因子
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim_per_head)
    
    # 计算注意力分数
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [batch_size, num_heads, seq_len_q, seq_len_k]
    
    # 应用softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
    
    # 计算加权和
    output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len_q, dim_per_head]
    
    return output

# 使用CUDA扩展的实现
def flash_attention_cuda(q, k, v, sm_scale=None):
    """
    使用CUDA扩展实现的Flash Attention
    输入形状: [batch_size, num_heads, seq_len_q, dim_per_head]
    输出形状: [batch_size, num_heads, seq_len_q, dim_per_head]
    """
    # 如果CUDA扩展不可用，使用回退实现
    if not CUDA_EXTENSION_AVAILABLE:
        return flash_attention_fallback(q, k, v, sm_scale)
    
    # 获取输入张量的维度
    batch_size, num_heads, seq_len_q, dim_per_head = q.shape
    
    # 设置缩放因子
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim_per_head)
    
    # 调用CUDA扩展
    output = flash_attention_cuda_ext.flash_attention(q, k, v, sm_scale)[0]
    
    return output

class FlashAttention(nn.Module):
    """
    使用CUDA实现的Flash Attention模块
    该模块设计用于替代标准的MultiHeadAttention模块，提供更高的性能
    """
    def __init__(self, d_model, num_heads):
        super(FlashAttention, self).__init__()
        
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
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播函数，与标准的MultiHeadAttention兼容
        """
        batch_size = q.size(0)
        
        # 获取序列长度
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        # 线性变换并分割为多头
        Q = self.W_q(q).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, d_k]
        K = self.W_k(k).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, d_k]
        V = self.W_v(v).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, d_k]
        
        # 计算Flash Attention
        # 注意：当前实现暂时忽略掩码，完整实现应该处理掩码
        attn_output = flash_attention_cuda(Q, K, V)
        
        # 由于flash_attention_cuda不返回注意力权重，我们创建一个假的权重张量
        attn_weights = torch.zeros(batch_size, self.num_heads, seq_len_q, seq_len_k, device=q.device)
        
        # 连接多头并通过输出层
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.W_o(attn_output)  # [batch_size, seq_len_q, d_model]
        
        return output, attn_weights

class CUDAMultiHeadAttention(nn.Module):
    """
    兼容标准PyTorch MultiHeadAttention接口的封装器
    使用CUDA实现的Flash Attention
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CUDAMultiHeadAttention, self).__init__()
        
        # 初始化Flash Attention模块
        self.flash_attention = FlashAttention(d_model, num_heads)
        
        # 存储参数以保持接口兼容性
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        前向传播函数，与标准的MultiHeadAttention接口兼容
        """
        # 调用Flash Attention
        output, attn_weights = self.flash_attention(query, key, value)
        
        # 应用dropout
        output = self.dropout(output)
        
        # 注意：当前实现暂时忽略掩码，完整实现应该处理掩码
        return output, attn_weights

# 编译CUDA扩展的函数
def compile_cuda_extension():
    """
    编译CUDA扩展
    这是一个辅助函数，实际使用时可能需要根据环境调整
    """
    import os
    import subprocess
    import sys
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义编译命令
    cmd = [
        sys.executable,
        '-m', 'torch.utils.cpp_extension',
        'compile',
        '--cxx', 'g++',
        '--nvcc', 'nvcc',
        '--verbose',
        os.path.join(current_dir, 'flash_attention_cuda_kernel.cu')
    ]
    
    try:
        # 执行编译命令
        subprocess.check_call(cmd)
        print("CUDA extension compiled successfully.")
        return True
    except Exception as e:
        print(f"Failed to compile CUDA extension: {e}")
        print("Please make sure you have CUDA and PyTorch installed correctly.")
        print("You can still use the fallback implementation.")
        return False

# 示例用法（如果直接运行此文件）
if __name__ == "__main__":
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 尝试编译CUDA扩展（可选）
    # compile_cuda_extension()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 初始化并测试Flash Attention
    flash_attn = FlashAttention(d_model, num_heads).to(device)
    
    # 测量性能
    import time
    
    # 预热
    for _ in range(5):
        output, _ = flash_attn(q, k, v)
    
    # 实际测量
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        output, _ = flash_attn(q, k, v)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    print(f"Flash Attention (CUDA) time: {(end_time - start_time) / 100 * 1000:.2f} ms per iteration")
    print(f"Output shape: {output.shape}")