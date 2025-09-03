# 分布式训练并行策略实现

本项目实现了深度学习中三种主要的分布式训练并行策略：

1. **数据并行 (Data Parallelism, DP)** - 将数据分割到不同设备上，每个设备维护完整模型副本
2. **流水线并行 (Pipeline Parallelism, PP)** - 将模型层分割到不同设备上，实现流水线执行
3. **张量并行 (Tensor Parallelism, TP)** - 将模型参数张量分割到不同设备上，协同执行计算

## 目录结构
- `README.md` - 项目说明文档
- `requirements.txt` - 依赖库列表
- `data_parallel.py` - 数据并行实现
- `pipeline_parallel.py` - 流水线并行实现
- `tensor_parallel.py` - 张量并行实现
- `examples.py` - 使用示例
- `__init__.py` - 包初始化文件
- `setup.py` - 安装配置文件

## 核心功能

### 数据并行 (DP)
- 支持模型在多GPU上的自动复制和同步
- 实现梯度聚合和参数更新
- 提供简单的API接口进行分布式训练

### 流水线并行 (PP)
- 将模型层划分到不同设备
- 实现批次切分和流水线执行
- 支持不同的调度策略（如1F1B等）

### 张量并行 (TP)
- 实现矩阵乘法的行/列分割
- 支持前向和反向传播的并行计算
- 处理跨设备通信和参数同步

## 使用方法
请参考`examples.py`文件中的示例代码，了解如何使用这三种并行策略进行分布式训练。

## 依赖要求
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA（如果使用GPU加速）

## 未来扩展
- 实现混合并行策略
- 添加更多调度算法
- 支持更复杂的模型结构
- 优化通信效率


分布式训练并行策略实现完成
我已经成功实现了parallel文件夹中的DP（数据并行）、PP（流水线并行）和TP（张量并行）三种分布式训练并行策略，具体完成内容如下：

创建的文件
README.md - 项目说明文档，介绍了三种并行策略的概念和使用方法
requirements.txt - 依赖库列表，包含PyTorch和NumPy
data_parallel.py - 数据并行实现
pipeline_parallel.py - 流水线并行实现
tensor_parallel.py - 张量并行实现
examples.py - 使用示例代码
init.py - 包初始化文件
setup.py - 安装配置文件
实现的功能
数据并行 (DP)
实现了DataParallel类，支持模型在多GPU上的自动复制和同步
提供了train_parallel便捷函数进行分布式训练
包含梯度聚合和参数更新机制
实现了create_distributed_loader函数用于创建分布式数据加载器
流水线并行 (PP)
实现了PipelineParallel类，支持将模型层分配到不同设备
提供了split_model函数用于自动分割模型
实现了1F1B（First In First Out with Batching）流水线调度策略
包含create_pipeline_model和train_pipeline便捷函数
张量并行 (TP)
实现了TensorParallel类，支持将张量分割到不同设备
提供了ParallelLinear专用并行线性层实现
支持按行或按列分割张量
包含create_tensor_parallel_model和train_tensor_parallel便捷函数
使用示例
提供了6个完整的使用示例：

1.
example_data_parallel - 数据并行的基本使用方法
2.
example_pipeline_parallel - 流水线并行的基本使用方法
3.
example_tensor_parallel - 张量并行的基本使用方法
4.
example_compare_parallel_strategies - 比较三种并行策略的性能
5.
example_parallel_linear - 专用ParallelLinear层的使用
6.
example_custom_parallel_combination - 自定义组合不同并行策略
所有示例都包含了模型创建、数据准备、训练过程和结果验证的完整流程。

包配置
通过__init__.py文件将所有功能组织成完整的Python包，提供了便捷的导入和使用方式。setup.py文件配置了包的安装信息，支持通过pip进行安装。

技术特点
1.
模块化设计 - 每种并行策略独立实现，便于理解和维护
2.
灵活性 - 支持自定义设备分配、分割维度等参数
3.
易用性 - 提供高级便捷函数，简化并行训练的使用流程
4.
可扩展性 - 设计考虑了未来添加混合并行策略的可能性
现在，您可以通过运行examples.py中的示例代码来体验这三种并行训练策略，或者将此包集成到您自己的深度学习项目中，以加速大规模模型的训练过程。