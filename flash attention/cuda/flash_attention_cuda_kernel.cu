#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// 常量定义
#define WARP_SIZE 32
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 64

// CUDA核函数：Flash Attention 实现
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,  // 查询矩阵 [batch_size, num_heads, seq_len_q, dim_per_head]
    const float* __restrict__ K,  // 键矩阵 [batch_size, num_heads, seq_len_k, dim_per_head]
    const float* __restrict__ V,  // 值矩阵 [batch_size, num_heads, seq_len_k, dim_per_head]
    float* __restrict__ O,        // 输出矩阵 [batch_size, num_heads, seq_len_q, dim_per_head]
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int dim_per_head,
    float sm_scale
) {
    // 获取当前线程在块中的索引
    int bid_m = blockIdx.x;
    int bid_n = blockIdx.y;
    int bid_b = blockIdx.z / num_heads;
    int bid_h = blockIdx.z % num_heads;
    
    // 计算当前批次和头的起始位置
    int batch_offset = bid_b * seq_len_q * seq_len_k * dim_per_head * num_heads;
    int head_offset = bid_h * seq_len_q * seq_len_k * dim_per_head;
    
    // 计算查询、键、值的起始指针
    const float* Q_ptr = Q + bid_b * num_heads * seq_len_q * dim_per_head + bid_h * seq_len_q * dim_per_head;
    const float* K_ptr = K + bid_b * num_heads * seq_len_k * dim_per_head + bid_h * seq_len_k * dim_per_head;
    const float* V_ptr = V + bid_b * num_heads * seq_len_k * dim_per_head + bid_h * seq_len_k * dim_per_head;
    float* O_ptr = O + bid_b * num_heads * seq_len_q * dim_per_head + bid_h * seq_len_q * dim_per_head;
    
    // 初始化共享内存用于存储查询、键和值的块
    __shared__ float sQ[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float sK[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ float sV[BLOCK_SIZE_N][BLOCK_SIZE_K];
    
    // 初始化累加器和掩码
    float acc[BLOCK_SIZE_K];
    for (int i = 0; i < BLOCK_SIZE_K; ++i) {
        acc[i] = 0.0f;
    }
    
    // 初始化注意力分数和softmax分母
    float attn_scores[BLOCK_SIZE_N];
    float max_attn = -INFINITY;
    float sum_attn = 0.0f;
    
    // 计算全局查询和键的起始位置
    int global_m = bid_m * BLOCK_SIZE_M + threadIdx.x;
    int global_n = bid_n * BLOCK_SIZE_N + threadIdx.y;
    
    // 迭代处理所有的键值块
    for (int k_offset = 0; k_offset < dim_per_head; k_offset += BLOCK_SIZE_K) {
        // 加载查询和键的块到共享内存
        if (global_m < seq_len_q && k_offset + threadIdx.y < dim_per_head) {
            sQ[threadIdx.x][threadIdx.y] = Q_ptr[global_m * dim_per_head + (k_offset + threadIdx.y)];
        } else {
            sQ[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        if (global_n < seq_len_k && k_offset + threadIdx.x < dim_per_head) {
            sK[threadIdx.x][threadIdx.y] = K_ptr[global_n * dim_per_head + (k_offset + threadIdx.x)];
        } else {
            sK[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        // 同步以确保共享内存中的数据已准备好
        __syncthreads();
        
        // 计算注意力分数
        if (global_m < seq_len_q && global_n < seq_len_k) {
            float score = 0.0f;
            for (int k = 0; k < BLOCK_SIZE_K; ++k) {
                score += sQ[threadIdx.x][k] * sK[k][threadIdx.y];
            }
            score *= sm_scale;
            
            // 计算softmax
            if (score > max_attn) {
                max_attn = score;
            }
            attn_scores[threadIdx.y] = score;
        }
        
        // 同步以确保分数计算完成
        __syncthreads();
        
        // 加载值的块到共享内存
        if (global_n < seq_len_k && k_offset + threadIdx.y < dim_per_head) {
            sV[threadIdx.x][threadIdx.y] = V_ptr[global_n * dim_per_head + (k_offset + threadIdx.y)];
        } else {
            sV[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        // 同步以确保值的块已加载
        __syncthreads();
        
        // 应用softmax并计算加权和
        if (global_m < seq_len_q && global_n < seq_len_k) {
            // 计算softmax
            float prob = expf(attn_scores[threadIdx.y] - max_attn);
            sum_attn += prob;
            
            // 计算加权和
            for (int k = 0; k < BLOCK_SIZE_K; ++k) {
                acc[k] += prob * sV[threadIdx.y][k];
            }
        }
        
        // 同步以确保当前迭代完成
        __syncthreads();
    }
    
    // 归一化并存储结果
    if (global_m < seq_len_q && global_n < seq_len_k) {
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            if (k < dim_per_head) {
                O_ptr[global_m * dim_per_head + k] = acc[k] / sum_attn;
            }
        }
    }
}

// Python 调用接口
std::vector<torch::Tensor> flash_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float sm_scale
) {
    // 检查输入设备是否为CUDA
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    // 获取输入张量的维度
    auto sizes = Q.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len_q = sizes[2];
    int dim_per_head = sizes[3];
    int seq_len_k = K.size(2);
    
    // 确保维度匹配
    TORCH_CHECK(K.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads, "Number of heads mismatch");
    TORCH_CHECK(K.size(3) == dim_per_head, "Dimension per head mismatch");
    TORCH_CHECK(V.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(V.size(1) == num_heads, "Number of heads mismatch");
    TORCH_CHECK(V.size(2) == seq_len_k, "Sequence length mismatch");
    TORCH_CHECK(V.size(3) == dim_per_head, "Dimension per head mismatch");
    
    // 初始化输出张量
    auto O = torch::empty_like(Q);
    
    // 设置CUDA核的网格和块大小
    dim3 block_size(BLOCK_SIZE_M, BLOCK_SIZE_N);
    dim3 grid_size(
        (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        (seq_len_k + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        batch_size * num_heads
    );
    
    // 启动CUDA核
    flash_attention_kernel<<<grid_size, block_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len_q,
        seq_len_k,
        dim_per_head,
        sm_scale
    );
    
    // 检查CUDA错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error during kernel execution");
    
    return {O};
}

// 注册Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention_cuda, "Flash Attention (CUDA)");
}