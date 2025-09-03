# Transformer 实现

这是一个基于PyTorch的Transformer模型完整实现，参考了"Attention is All You Need"论文中的架构设计。本实现采用模块化设计，便于理解和扩展。

## 项目结构

```
transformer_implementation/
├── requirements.txt    # 项目依赖
├── multi_head_attention.py  # 多头注意力机制实现
├── feed_forward.py     # 前馈网络实现
├── positional_encoding.py  # 位置编码实现
├── encoder_layer.py    # 编码器层实现
├── decoder_layer.py    # 解码器层实现
├── encoder.py          # 编码器实现
├── decoder.py          # 解码器实现
├── transformer.py      # 完整Transformer模型实现
├── utils.py            # 工具函数
├── example.py          # 使用示例
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

## 使用示例

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

## 自定义和扩展

### 调整模型参数

你可以根据自己的需求调整以下参数：
- `d_model`: 模型维度
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