import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from data_utils import Vocabulary

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
    """计算BLEU分数以评估翻译质量"""
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    
    try:
        # 将参考和候选翻译分词
        reference = [word_tokenize(ref) for ref in reference]
        candidate = word_tokenize(candidate)
        
        # 计算不同n-gram的权重
        weights = tuple(1.0/n_gram for _ in range(n_gram))
        
        # 计算BLEU分数
        score = sentence_bleu(reference, candidate, weights=weights)
        return score
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0

# 可视化训练过程中的损失和BLEU分数
def plot_training_progress(train_losses, val_losses, val_bleu_scores=None, save_path=None):
    """绘制训练和验证损失曲线以及验证BLEU分数曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制BLEU分数曲线（如果提供）
    if val_bleu_scores is not None:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(val_bleu_scores) + 1), val_bleu_scores, label='Validation BLEU', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('Validation BLEU Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # BLEU分数范围在0到1之间
    
    plt.tight_layout()
    
    # 保存图像（如果提供了保存路径）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# 保存词汇表
def save_vocabulary(vocab, file_path):
    """将词汇表保存到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for idx, token in vocab.itos.items():
            f.write(f'{token}\t{idx}\n')

# 加载词汇表
def load_vocabulary(file_path):
    """从文件加载词汇表"""
    vocab = Vocabulary()
    vocab.itos = {}
    vocab.stoi = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            token, idx = parts
            idx = int(idx)
            vocab.itos[idx] = token
            vocab.stoi[token] = idx
    
    vocab.vocab_size = len(vocab.itos)
    return vocab