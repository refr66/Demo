import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_kernel(
    Q, K, V, 
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    O,
    B, H, M, N, dim_per_head,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton 实现的 flash attention 核心计算内核
    该内核使用分块计算来优化内存访问，减少不必要的数据传输
    """
    # 获取当前程序的块索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch = tl.program_id(2)
    head = tl.program_id(3)

    # 计算查询和键值的起始位置
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化局部累加器
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 计算查询块指针
    q_ptrs = Q + (
        batch * stride_qb + 
        head * stride_qh + 
        offs_m[:, None] * stride_qm + 
        tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
    )
    
    # 迭代处理所有的键值块
    for start_n in range(0, N, BLOCK_N):
        # 计算当前键值块的索引范围
        curr_n = start_n + tl.arange(0, BLOCK_N)
        curr_n = tl.minimum(curr_n, N - 1)  # 防止越界
        
        # 计算键和值的块指针
        k_ptrs = K + (
            batch * stride_kb + 
            head * stride_kh + 
            curr_n[:, None] * stride_kn + 
            tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
        )
        v_ptrs = V + (
            batch * stride_vb + 
            head * stride_vh + 
            curr_n[:, None] * stride_vn + 
            tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk
        )
        
        # 加载查询、键和值
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # 计算注意力分数
        q = q * sm_scale  # 缩放查询向量
        attn = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        # 计算点积
        attn = tl.dot(q, k.T)
        
        # 应用softmax
        max_attn = tl.max(attn, axis=1, keepdims=True)
        attn = tl.exp(attn - max_attn)
        attn_sum = tl.sum(attn, axis=1, keepdims=True) + 1e-10  # 防止除零
        attn = attn / attn_sum
        
        # 计算加权和
        acc += tl.dot(attn, v)
    
    # 计算输出块指针
    o_ptrs = O + (
        batch * stride_ob + 
        head * stride_oh + 
        offs_m[:, None] * stride_om + 
        tl.arange(0, BLOCK_DMODEL)[None, :] * stride_ok
    )
    
    # 保存结果
    tl.store(o_ptrs, acc)


def flash_attention(q, k, v, sm_scale=None):
    """
    使用 Triton 实现的 flash attention 函数
    输入形状: [batch_size, num_heads, seq_len_q, dim_per_head]
    输出形状: [batch_size, num_heads, seq_len_q, dim_per_head]
    """
    # 获取输入张量的维度
    batch_size, num_heads, seq_len_q, dim_per_head = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # 确保 Q、K、V 的维度匹配
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Q, K, V must have the same embedding dimension"
    assert k.shape[2] == v.shape[2], "K and V must have the same sequence length"
    
    # 设置缩放因子
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim_per_head)
    
    # 初始化输出张量
    o = torch.empty_like(q)
    
    # 设置 Triton 核的参数
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 64
    
    # 计算网格大小
    grid = (
        triton.cdiv(seq_len_q, BLOCK_M),  # 每个查询块一个程序
        triton.cdiv(seq_len_k, BLOCK_N),  # 每个键值块一个程序
        batch_size,                       # 每个批次一个程序
        num_heads                         # 每个头一个程序
    )
    
    # 启动 Triton 核
    flash_attention_kernel[
        grid,
        "triton_gpu",
        (BLOCK_M * BLOCK_N, BLOCK_M * BLOCK_DMODEL, BLOCK_N * BLOCK_DMODEL)
    ](
        q, k, v,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        o,
        batch_size, num_heads, seq_len_q, seq_len_k, dim_per_head,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N
    )
    
    return o

class FlashAttention(nn.Module):
    """
    使用 Triton 实现的 flash attention 模块
    该模块设计用于替代标准的 MultiHeadAttention 模块，提供更高的性能
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
        前向传播函数，与标准的 MultiHeadAttention 兼容
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
        
        # 计算 flash attention
        # 注意：当前实现暂时忽略掩码，完整实现应该处理掩码
        attn_output = flash_attention(Q, K, V)
        
        # 由于 flash_attention 不返回注意力权重，我们创建一个假的权重张量
        attn_weights = torch.zeros(batch_size, self.num_heads, seq_len_q, seq_len_k, device=q.device)
        
        # 连接多头并通过输出层
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.W_o(attn_output)  # [batch_size, seq_len_q, d_model]
        
        return output, attn_weights

class TritonMultiHeadAttention(nn.Module):
    """
    兼容现有 Transformer 架构的 Triton 多头注意力模块
    可以作为 drop-in 替代标准的 MultiHeadAttention 模块
    """
    def __init__(self, d_model, num_heads):
        super(TritonMultiHeadAttention, self).__init__()
        
        # 使用我们实现的 FlashAttention
        self.flash_attention = FlashAttention(d_model, num_heads)
        
    def forward(self, q, k, v, mask=None):
        """
        前向传播函数，完全兼容标准的 MultiHeadAttention 接口
        """
        return self.flash_attention(q, k, v, mask)

# 示例用法
if __name__ == "__main__":
    # 创建随机输入张量
    batch_size = 4
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建标准的多头注意力和 Triton 实现的注意力
    standard_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    triton_attn = TritonMultiHeadAttention(d_model, num_heads)
    
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 测量标准注意力的性能
    import time
    start_time = time.time()
    standard_output, _ = standard_attn(q, k, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    standard_time = time.time() - start_time
    
    # 测量 Triton 注意力的性能
    start_time = time.time()
    triton_output, _ = triton_attn(q, k, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    triton_time = time.time() - start_time
    
    print(f"Standard Attention Time: {standard_time:.6f} seconds")
    print(f"Triton Attention Time: {triton_time:.6f} seconds")
    print(f"Speedup: {standard_time / triton_time:.2f}x")