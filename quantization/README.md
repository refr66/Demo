# Post-Training Quantization (PTQ) 工具

一个简单的Post-Training Quantization (PTQ)工具，用于将训练好的深度学习模型（如Transformer）的FP32/FP16权重转换为INT8，实现模型压缩和加速。

## 功能概述

- **校准 (Calibration)**: 向模型输入少量样本数据，记录每一层激活值的动态范围（最大最小值）
- **缩放因子计算**: 根据动态范围计算从浮点数到INT8的映射关系
- **权重量化**: 将模型的权重从浮点数转换为INT8整数
- **INT8推理模拟**: 实现简单的INT8矩阵乘法，展示伪量化和真量化的区别
- **精度评估**: 评估量化后模型的精度损失

## 目录结构

```
quantization/
├── README.md              # 项目说明文档
├── requirements.txt       # 项目依赖
├── quantization.py        # 核心量化功能实现
├── calibration.py         # 校准功能实现
├── inference.py           # INT8推理逻辑实现
├── examples.py            # 使用示例
├── __init__.py            # 包定义文件
└── setup.py               # 安装配置文件
```

## 核心概念

### 量化原理

量化是将高精度浮点数（FP32/FP16）表示转换为低精度整数（INT8）表示的过程。主要步骤包括：

1. **确定量化范围**: 通过校准数据确定激活值和权重的动态范围
2. **计算缩放因子**: 缩放因子 = (最大值 - 最小值) / (量化范围最大值 - 量化范围最小值)
3. **零点调整**: 确定偏移量，确保零点能够被精确表示
4. **量化转换**: 将浮点数映射到整数域
5. **反量化**: 在推理时将整数结果转换回浮点数

### 量化类型

- **对称量化**: 零点固定为0，仅使用缩放因子
- **非对称量化**: 同时使用缩放因子和零点偏移
- **伪量化**: 在浮点计算中模拟量化效果，用于QAT
- **真量化**: 实际使用整数运算

## 使用方法

### 基本使用

```python
import torch
from quantization import quantize_model

# 加载训练好的模型
model = torch.load('path/to/model.pt')

# 准备校准数据
calibration_data = load_calibration_data()

# 量化模型
quantized_model = quantize_model(model, calibration_data)

# 保存量化模型
save_quantized_model(quantized_model, 'path/to/quantized_model.pt')

# 使用量化模型进行推理
input_data = torch.randn(1, 10, 512)
output = quantized_model(input_data)
```

### 自定义量化参数

```python
from quantization import PTQQuantizer

# 创建量化器实例
quantizer = PTQQuantizer(
    bits=8,              # 量化位宽
    symmetric=True,      # 是否使用对称量化
    calibration_method='minmax'  # 校准方法
)

# 执行量化
quantized_model = quantizer.quantize(model, calibration_data)
```

## 面试价值

- **模型压缩和加速**: 证明你理解深度学习模型优化的关键技术
- **数值表示**: 展示你对计算机数值表示（浮点数 vs 定点数）有深入认识
- **精度优化**: 可以讨论量化带来的精度损失，以及如何通过QAT等技术来弥补
- **部署优化**: 理解模型从训练到部署的转换过程

## 性能考虑

量化可以带来以下好处：

- **内存占用减少**: INT8量化可将模型大小减少约75%
- **计算加速**: 支持INT8指令集的硬件（如Intel AVX2、ARM NEON）可提供显著性能提升
- **功耗降低**: 低精度计算通常消耗更少的能量

## 未来扩展

- 支持更多量化位宽（如4位、2位量化）
- 实现量化感知训练(QAT)
- 支持更复杂的量化策略（如通道量化、分组量化）
- 提供与主流推理框架的集成（如ONNX Runtime、TensorRT）



Post-Training Quantization (PTQ) 工具实现总结
我已经完成了一个简单而完整的Post-Training Quantization (PTQ)工具的实现，该工具能够将训练好的深度学习模型（如Transformer）的FP32/FP16权重转换为INT8表示，实现模型压缩和加速。

创建的文件
1.
README.md - 详细描述了PTQ工具的功能概述、核心概念、使用方法和面试价值
2.
requirements.txt - 列出了项目依赖（torch>=2.0.0和numpy>=1.20.0）
3.
quantization.py - 实现了核心量化功能，包括Quantizer基类和PTQQuantizer类
4.
calibration.py - 实现了模型校准功能，支持多种校准方法
5.
inference.py - 实现了INT8推理逻辑，包括伪量化和真量化的实现
6.
examples.py - 提供了8个使用示例，展示工具的各种功能
7.
init.py - 将所有模块组织成完整的包，导出核心功能
8.
setup.py - 定义了包安装配置，便于用户安装使用
实现的功能
1. 核心量化功能
量化器实现：支持对称和非对称量化策略
权重量化：将FP32/FP16权重转换为INT8整数
缩放因子计算：根据动态范围计算从浮点数到INT8的映射关系
零点调整：支持零点偏移量计算，确保零点能够被精确表示
2. 校准功能
多种校准方法：支持minmax、percentile和histogram三种校准方法
统计信息收集：通过前向钩子收集各层激活值的统计信息
量化范围计算：根据收集的统计信息确定合适的量化范围
异常值处理：通过percentile和histogram方法有效处理异常值
3. INT8推理功能
伪量化实现：在浮点计算中模拟量化效果
真量化支持：实现了INT8矩阵乘法，展示了真量化计算过程
模型转换：将标准PyTorch模型转换为支持INT8权重的量化模型
精度评估：提供了量化前后模型精度的比较功能
4. 实用工具
张量量化：支持单独对张量进行量化和反量化操作
模型保存与加载：支持量化模型的保存和加载
性能比较：提供了量化前后的模型大小和精度比较
技术亮点
1.
完整的量化流程：实现了从校准、量化到推理的完整流程
2.
灵活的量化策略：支持对称/非对称量化、不同位宽和校准方法
3.
精确的INT8计算：正确实现了考虑零点偏移的INT8矩阵乘法
4.
模块化设计：清晰的代码结构，便于扩展和维护
5.
丰富的使用示例：提供了8个不同场景的使用示例，覆盖了工具的所有功能
使用方法
PTQ工具提供了简洁直观的API，用户可以轻松地对模型进行量化：

```python
import torch
from quantization import quantize_model

# 加载训练好的模型
model = torch.load('path/to/model.pt')

# 准备校准数据
calibration_data = [torch.randn(32, input_dim) for _ in range(10)]

# 量化模型
quantized_model = quantize_model(model, calibration_data)

# 使用量化模型进行推理
input_data = torch.randn(1, input_dim)
output = quantized_model(input_data)
```
应用价值
这个PTQ工具实现了模型压缩和加速的关键技术，展示了对计算机数值表示的深入理解，以及对量化带来的精度损失的处理方法。通过使用该工具，可以将深度学习模型大小减少约75%，并在支持INT8指令集的硬件上获得显著的性能提升。

该项目可以作为学习模型量化技术的教学工具，也可以作为更复杂量化系统开发的起点。通过进一步扩展，可以支持量化感知训练(QAT)、更复杂的量化策略和与主流推理框架的集成。