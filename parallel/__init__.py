"""
分布式训练并行策略包

本包提供了三种主要的分布式训练并行策略实现：
1. 数据并行 (Data Parallelism, DP)
2. 流水线并行 (Pipeline Parallelism, PP)
3. 张量并行 (Tensor Parallelism, TP)

每个并行策略都提供了完整的实现，包括模型并行化、前向/反向传播和参数更新等功能。
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "AI Sys Team"
__description__ = "分布式训练并行策略实现包"

# 导出核心类和函数
from .data_parallel import (
    DataParallel,
    train_parallel,
    create_distributed_loader,
    copy_model
)

from .pipeline_parallel import (
    PipelineParallel,
    create_pipeline_model,
    train_pipeline,
    split_model
)

from .tensor_parallel import (
    TensorParallel,
    create_tensor_parallel_model,
    train_tensor_parallel,
    ParallelLinear
)

from .examples import (
    example_data_parallel,
    example_pipeline_parallel,
    example_tensor_parallel,
    example_compare_parallel_strategies,
    example_parallel_linear,
    example_custom_parallel_combination,
    run_all_examples,
    SimpleDataset,
    SimpleModel,
    DeepModel
)

# 便捷访问函数
def get_data_parallel(model, device_ids=None, backend='gloo'):
    """
    获取数据并行模型实例
    
    参数:
        model: 要并行化的模型
        device_ids: 设备ID列表
        backend: 分布式通信后端
    
    返回:
        DataParallel实例
    """
    return DataParallel(model, device_ids, backend)

def get_pipeline_parallel(model_layers, device_ids, chunk_size=1):
    """
    获取流水线并行模型实例
    
    参数:
        model_layers: 模型层列表
        device_ids: 设备ID列表
        chunk_size: 流水线批处理块大小
    
    返回:
        PipelineParallel实例
    """
    return PipelineParallel(model_layers, device_ids, chunk_size)

def get_tensor_parallel(model, device_ids=None, split_dim=0, backend='gloo'):
    """
    获取张量并行模型实例
    
    参数:
        model: 要并行化的模型
        device_ids: 设备ID列表
        split_dim: 张量分割维度
        backend: 分布式通信后端
    
    返回:
        TensorParallel实例
    """
    return create_tensor_parallel_model(model, device_ids, split_dim, backend)

# 包描述
__doc__ = """
分布式训练并行策略包

本包实现了深度学习中三种主要的分布式训练并行策略：

## 数据并行 (Data Parallelism)
将训练数据分割到不同的设备上，每个设备维护完整的模型副本，通过梯度聚合和参数同步实现并行训练。
适用于模型较小但数据量较大的场景。

主要组件：
- DataParallel: 数据并行模型包装器
- train_parallel: 数据并行训练函数
- create_distributed_loader: 创建分布式数据加载器

## 流水线并行 (Pipeline Parallelism)
将模型的不同层分配到不同的设备上，通过流水线执行方式实现并行训练。
适用于模型较大、无法在单个设备上容纳的场景。

主要组件：
- PipelineParallel: 流水线并行模型包装器
- create_pipeline_model: 创建流水线并行模型
- train_pipeline: 流水线并行训练函数
- split_model: 模型分割函数

## 张量并行 (Tensor Parallelism)
将模型的大型张量（如权重矩阵）分割到不同的设备上，通过协同计算实现并行训练。
适用于模型中包含非常大的张量的场景。

主要组件：
- TensorParallel: 张量并行模型包装器
- create_tensor_parallel_model: 创建张量并行模型
- train_tensor_parallel: 张量并行训练函数
- ParallelLinear: 并行线性层实现

## 使用示例
包中包含多个使用示例，展示了如何使用这三种并行策略进行分布式训练：
- example_data_parallel: 数据并行使用示例
- example_pipeline_parallel: 流水线并行使用示例
- example_tensor_parallel: 张量并行使用示例
- example_compare_parallel_strategies: 比较三种并行策略的性能
- example_parallel_linear: 使用ParallelLinear层的示例
- example_custom_parallel_combination: 自定义组合并行策略的示例
- run_all_examples: 运行所有示例

## 依赖要求
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA（如果使用GPU加速）
"""

# 公共API列表
__all__ = [
    # 数据并行相关
    'DataParallel',
    'train_parallel',
    'create_distributed_loader',
    'copy_model',
    
    # 流水线并行相关
    'PipelineParallel',
    'create_pipeline_model',
    'train_pipeline',
    'split_model',
    
    # 张量并行相关
    'TensorParallel',
    'create_tensor_parallel_model',
    'train_tensor_parallel',
    'ParallelLinear',
    
    # 示例相关
    'example_data_parallel',
    'example_pipeline_parallel',
    'example_tensor_parallel',
    'example_compare_parallel_strategies',
    'example_parallel_linear',
    'example_custom_parallel_combination',
    'run_all_examples',
    'SimpleDataset',
    'SimpleModel',
    'DeepModel',
    
    # 便捷函数
    'get_data_parallel',
    'get_pipeline_parallel',
    'get_tensor_parallel',
    
    # 版本信息
    '__version__',
    '__author__',
    '__description__'
]