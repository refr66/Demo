# Transformer 实现

这是一个基于PyTorch的Transformer模型完整实现，参考了"Attention is All You Need"论文中的架构设计。本实现采用模块化设计，包含了从数据处理到模型训练再到推理的完整流程，便于理解和扩展。

## 项目结构

```
transformer_implementation/
├── requirements.txt    # 项目依赖
├── data_utils.py       # 数据处理和词汇表构建
├── multi_head_attention.py  # 多头注意力机制实现
├── feed_forward.py     # 前馈网络实现
├── positional_encoding.py  # 位置编码实现
├── encoder_layer.py    # 编码器层实现
├── decoder_layer.py    # 解码器层实现
├── encoder.py          # 编码器实现
├── decoder.py          # 解码器实现
├── transformer.py      # 完整Transformer模型实现
├── train.py            # 模型训练脚本
├── inference.py        # 模型推理和翻译脚本
├── utils.py            # 工具函数
├── example.py          # 简单使用示例
└── README.md           # 项目说明
```

## 实现特点

1. **模块化设计**：每个组件独立实现，便于理解和维护
2. **Pre-LN架构**：采用更现代的Pre-Layer Normalization设计，训练更稳定
3. **完整功能**：包含掩码机制、位置编码、多头注意力等所有Transformer核心功能
4. **详细注释**：代码中包含详细注释，便于学习和理解
5. **辅助工具**：提供了注意力可视化、模型参数计算等实用工具

## 核心组件说明

### 1. 多头注意力机制 (Multi-Head Attention)
- 实现了缩放点积注意力计算
- 支持多头并行计算和掩码机制
- 包含Q、K、V线性变换和输出线性变换

### 2. 前馈网络 (Feed Forward Network)
- 两层线性变换结构
- 使用GELU激活函数，比ReLU有更好的性能
- 包含Dropout层防止过拟合

### 3. 位置编码 (Positional Encoding)
- 使用正弦和余弦函数实现的位置编码
- 为模型提供序列的位置信息
- 支持任意长度的序列

### 4. 编码器层 (Encoder Layer)
- 包含自注意力子层
- 包含前馈网络子层
- 采用Pre-LN架构，在进入子层前进行层归一化
- 包含残差连接和Dropout层

### 5. 解码器层 (Decoder Layer)
- 包含掩码自注意力子层
- 包含编码器-解码器交叉注意力子层
- 包含前馈网络子层
- 同样采用Pre-LN架构

### 6. 编码器 (Encoder)
- 由多个编码器层堆叠而成
- 包含词嵌入层和位置编码层
- 输出层归一化

### 7. 解码器 (Decoder)
- 由多个解码器层堆叠而成
- 包含词嵌入层和位置编码层
- 输出层归一化

### 8. 完整Transformer模型
- 结合编码器和解码器
- 包含输出层，将解码器输出映射到目标词汇表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 完整功能

本项目实现了Transformer的完整功能流程：

1. **数据处理**：包含词汇表构建、分词器、数据集处理和批量处理
2. **模型训练**：支持完整的训练循环、验证评估、模型保存和加载
3. **模型推理**：支持贪心解码和束搜索解码算法
4. **评估指标**：集成了BLEU分数计算来评估翻译质量
5. **可视化工具**：支持注意力权重可视化和训练过程可视化

## 使用指南

### 1. 基本示例

可以通过运行`example.py`来查看模型的基本使用方法：

```bash
python example.py
```

该示例演示了：
- 创建Transformer模型
- 生成随机输入数据
- 创建掩码
- 前向传播计算
- 查看输出结果

### 2. 训练模型

使用`train.py`来训练一个完整的Transformer翻译模型：

```bash
python train.py
```

训练脚本的主要功能：
- 自动下载Multi30k数据集（德语到英语翻译）
- 构建源语言和目标语言的词汇表
- 设置训练参数（批次大小、学习率、训练轮数等）
- 实现完整的训练循环，包含教师强制（Teacher Forcing）
- 实现验证循环，计算验证损失和BLEU分数
- 自动保存表现最好的模型

你可以在`train.py`文件中修改以下参数：
- `d_model`: 模型维度
- `num_heads`: 多头注意力的头数
- `num_layers`: 编码器和解码器的层数
- `d_ff`: 前馈网络内部层的维度
- `batch_size`: 训练批次大小
- `epochs`: 训练轮数
- `lr`: 学习率

### 3. 模型推理/翻译

使用`inference.py`来使用训练好的模型进行翻译：

```bash
python inference.py
```

推理脚本的主要功能：
- 加载训练好的模型权重
- 支持两种解码策略：贪心解码和束搜索解码
- 提供交互式翻译界面

## 核心模块详解

### 数据处理模块 (`data_utils.py`)

- `Vocabulary`类：负责构建和管理词汇表，包含词频统计、标记到索引的转换等功能
- `SimpleTokenizer`类：实现简单的文本分词功能
- `TranslationDataset`类：用于加载和处理翻译数据集
- `collate_fn`函数：处理变长序列的批量填充
- 数据集加载和词汇表构建的辅助函数

### 训练模块 (`train.py`)

- `Trainer`类：实现完整的训练和评估流程
- 支持学习率调度器（NoamOpt）
- 梯度裁剪防止梯度爆炸
- 模型保存和加载功能
- 支持BLEU分数评估

### 推理模块 (`inference.py`)

- `Translator`类：实现翻译功能，支持两种解码策略
- 贪心解码：每一步选择概率最高的词
- 束搜索解码：同时保留多个最可能的候选序列，提高翻译质量

### 工具函数 (`utils.py`)

- 掩码创建函数：为源序列和目标序列创建掩码
- 注意力可视化：绘制注意力权重热图
- 训练过程可视化：绘制损失曲线和BLEU分数曲线
- BLEU分数计算：评估翻译质量
- 词汇表保存和加载功能

## 自定义和扩展

### 调整模型参数

- `num_heads`: 多头注意力的头数
- `num_layers`: 编码器和解码器的层数
- `d_ff`: 前馈网络内部层的维度
- `dropout`: Dropout概率

### 用于实际任务

要将此模型用于实际任务，你需要：
1. 准备适合你任务的数据集
2. 实现数据预处理和tokenization
3. 添加训练循环，包括优化器和损失函数
4. 实现评估和推理功能

## 工具函数

`utils.py`中提供了以下实用工具：
- `create_src_mask`: 生成源序列掩码
- `create_tgt_mask`: 生成目标序列掩码
- `plot_attention_weights`: 可视化注意力权重
- `count_parameters`: 计算模型参数数量
- `initialize_weights`: 初始化模型权重
- `NoamOpt`: 学习率调度器
- `compute_bleu`: 计算BLEU分数

## 参考资料

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer的原始论文
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html) - PyTorch的官方文档
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Transformer的详细注释实现

## 注意事项

- 此实现采用Pre-LN架构，与原始论文中的Post-LN架构有所不同
- 此代码主要用于学习和理解Transformer的工作原理
- 在实际应用中，你可能需要根据具体任务进行调整和优化

## 许可证

[MIT License](https://opensource.org/licenses/MIT)