import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成源序列掩码（用于屏蔽填充位置）
def create_src_mask(src, pad_idx):
    # src的形状: [batch_size, src_len]
    # 创建掩码，填充位置为0，非填充位置为1
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
    return src_mask

# 生成目标序列掩码（包括填充掩码和前瞻掩码）
def create_tgt_mask(tgt, pad_idx):
    # tgt的形状: [batch_size, tgt_len]
    
    # 创建填充掩码
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
    
    # 创建前瞻掩码（上三角矩阵，对角线及以下为1，以上为0）
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device))  # [tgt_len, tgt_len]
    
    # 合并两种掩码
    tgt_mask = tgt_pad_mask & tgt_sub_mask.bool().unsqueeze(0).unsqueeze(1)  # [batch_size, 1, tgt_len, tgt_len]
    
    return tgt_mask

# 绘制注意力权重热图
def plot_attention_weights(attention, sentence, result, layer=0, head=0, src_vocab=None, tgt_vocab=None):
    # 获取指定层和头的注意力权重
    attn = attention[layer][0, head].cpu().detach().numpy()  # [tgt_len, src_len]
    
    # 准备句子
    if src_vocab is not None and tgt_vocab is not None:
        sentence = [src_vocab.itos[token] for token in sentence]
        result = [tgt_vocab.itos[token] for token in result]
    
    # 创建热力图
    plt.figure(figsize=(10, 10))
    plt.matshow(attn, cmap='viridis')
    plt.xticks(range(len(sentence)), sentence, rotation=90)
    plt.yticks(range(len(result)), result)
    plt.colorbar()
    plt.title(f'Attention weights (Layer {layer+1}, Head {head+1})')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()

# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 初始化模型权重
def initialize_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

# 学习率预热和衰减调度器
class NoamOpt:
    """学习率调度器：先预热，然后按步数的平方根衰减"""
    def __init__(self, d_model, factor, warmup_steps, optimizer):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5) * 
                           min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)))
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.factor * (self.d_model ** (-0.5) * 
                             min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)))

# 计算BLEU分数（简单实现）
def compute_bleu(reference, candidate, n_gram=4):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    
    reference = [word_tokenize(ref) for ref in reference]
    candidate = word_tokenize(candidate)
    
    weights = tuple(1.0/n_gram for _ in range(n_gram))
    return sentence_bleu(reference, candidate, weights=weights)