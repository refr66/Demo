import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time
import os
from tqdm import tqdm
import numpy as np
from transformer import Transformer
from data_utils import TranslationDataset, collate_fn, load_translation_dataset, build_vocab_and_tokenizer, PAD_TOKEN
from utils import NoamOpt, compute_bleu

class Trainer:
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # 获取特殊标记的索引
        self.pad_idx = src_vocab.stoi[PAD_TOKEN]
        
        # 定义损失函数，忽略填充标记
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        # 初始化优化器
        self.optimizer = None
        self.scheduler = None
        
        # 用于记录训练过程
        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_val_loss = float('inf')
        
    def setup_optimizer(self, lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98)):
        """设置优化器"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
    def setup_scheduler(self, d_model, warmup_steps=4000, factor=1.0):
        """设置学习率调度器（NoamOpt）"""
        if self.optimizer is None:
            self.setup_optimizer()
        self.scheduler = NoamOpt(d_model, factor, warmup_steps, self.optimizer)
        
    def train_epoch(self, train_loader, clip=1.0):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # 将数据移至设备
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Teacher Forcing: decoder的输入是真实目标序列，向右移动一位
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
            tgt_mask = self.create_tgt_mask(tgt_input)
            
            # 前向传播
            output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            
            # 更新权重
            if self.scheduler is not None:
                self.scheduler.step()
            else:
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        all_references = []
        all_candidates = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # 将数据移至设备
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Teacher Forcing
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # 创建掩码
                src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
                tgt_mask = self.create_tgt_mask(tgt_input)
                
                # 前向传播
                output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # 计算损失
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                total_loss += loss.item()
                
                # 计算BLEU分数（可选，可能会增加计算时间）
                batch_size = src.size(0)
                for i in range(batch_size):
                    # 获取真实序列和预测序列
                    tgt_sequence = tgt_output[i].cpu().tolist()
                    pred_sequence = torch.argmax(output[i], dim=-1).cpu().tolist()
                    
                    # 移除填充标记
                    tgt_sequence = [token for token in tgt_sequence if token != self.pad_idx]
                    pred_sequence = [token for token in pred_sequence if token != self.pad_idx]
                    
                    # 转换为文本
                    reference_text = ' '.join(self.tgt_vocab.denumericalize(tgt_sequence))
                    candidate_text = ' '.join(self.tgt_vocab.denumericalize(pred_sequence))
                    
                    all_references.append([reference_text])
                    all_candidates.append(candidate_text)
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        
        # 计算BLEU分数
        if all_references and all_candidates:
            bleu_scores = [compute_bleu(refs, cand) for refs, cand in zip(all_references, all_candidates)]
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
        else:
            avg_bleu = 0.0
        
        return avg_loss, avg_bleu
    
    def train(self, train_loader, val_loader, epochs=10, clip=1.0, save_dir='models'):
        """完整的训练过程"""
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录训练开始时间
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, clip)
            self.train_losses.append(train_loss)
            
            # 在验证集上评估
            val_loss, val_bleu = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(val_bleu)
            
            # 记录学习率（如果使用了调度器）
            lr = self.scheduler.get_lr() if self.scheduler is not None else self.optimizer.param_groups[0]['lr']
            
            # 打印epoch结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val BLEU: {val_bleu:.4f}")
            print(f"Learning Rate: {lr:.6f}")
            
            # 保存表现最好的模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_bleu_scores': self.val_bleu_scores
                }, model_path)
                print(f"Model saved to {model_path}")
        
        # 记录训练结束时间
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def create_tgt_mask(self, tgt):
        """创建目标序列的掩码（填充掩码和前瞻掩码）"""
        # 填充掩码
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 前瞻掩码
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device))
        
        # 合并掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool().unsqueeze(0).unsqueeze(1)
        
        return tgt_mask

# 主函数，用于快速启动训练
if __name__ == "__main__":
    # 模型参数
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 100
    
    # 训练参数
    batch_size = 32
    epochs = 20
    lr = 0.0001
    clip = 1.0
    
    # 数据集参数
    dataset_name = 'multi30k'
    src_lang = 'de'
    tgt_lang = 'en'
    
    # 加载训练数据和验证数据
    print("Loading datasets...")
    train_src_data, train_tgt_data = load_translation_dataset(dataset_name, 'train', src_lang, tgt_lang)
    val_src_data, val_tgt_data = load_translation_dataset(dataset_name, 'validation', src_lang, tgt_lang)
    
    # 构建词汇表
    print("Building vocabularies...")
    src_vocab, tgt_vocab = build_vocab_and_tokenizer(train_src_data, train_tgt_data, min_freq=2)
    
    # 创建数据集
    train_dataset = TranslationDataset(train_src_data, train_tgt_data, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_src_data, val_tgt_data, src_vocab, tgt_vocab, max_len)
    
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print("Creating model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len
    )
    
    # 创建训练器
    trainer = Trainer(model, src_vocab, tgt_vocab)
    
    # 设置优化器和调度器
    trainer.setup_optimizer(lr=lr)
    trainer.setup_scheduler(d_model=d_model, warmup_steps=4000)
    
    # 开始训练
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=epochs, clip=clip)