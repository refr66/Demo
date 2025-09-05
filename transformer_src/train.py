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
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda' if torch.cuda.is_available() else 'cpu', use_fp16=False, grad_accumulation_steps=1):
        # 确保device是torch.device对象
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.use_fp16 = use_fp16
        self.grad_accumulation_steps = grad_accumulation_steps
        
        # 混合精度训练的scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 and self.device.type == 'cuda' else None
        
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
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training", mininterval=0.5)):
            # 将数据移至设备
            src = batch['src'].to(self.device, non_blocking=True)
            tgt = batch['tgt'].to(self.device, non_blocking=True)
            
            # Teacher Forcing: decoder的输入是真实目标序列，向右移动一位
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
            tgt_mask = self.create_tgt_mask(tgt_input)
            
            # 使用混合精度训练
            if self.use_fp16 and self.scaler is not None:
                with torch.amp.autocast(device_type=self.device.type):
                    # 前向传播
                    output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                    
                    # 计算损失
                    loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                    
                    # 梯度累积
                    loss = loss / self.grad_accumulation_steps
                    
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪和参数更新
                if (i + 1) % self.grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    if clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                # 标准精度训练
                # 前向传播
                output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # 计算损失
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                
                # 梯度累积
                loss = loss / self.grad_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪和参数更新
                if (i + 1) % self.grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * self.grad_accumulation_steps
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, compute_bleu_score=False):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        all_references = []
        all_candidates = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", mininterval=0.5):
                # 将数据移至设备
                src = batch['src'].to(self.device, non_blocking=True)
                tgt = batch['tgt'].to(self.device, non_blocking=True)
                
                # Teacher Forcing
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # 创建掩码
                src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
                tgt_mask = self.create_tgt_mask(tgt_input)
                
                # 前向传播
                if self.use_fp16 and self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type):
                        output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                else:
                    output, _, _, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # 计算损失
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                total_loss += loss.item()
                
                # 计算BLEU分数（可选，每N个批次计算一次或完全跳过）
                if compute_bleu_score:
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
        avg_bleu = 0.0
        if compute_bleu_score and all_references and all_candidates:
            bleu_scores = [compute_bleu(refs, cand) for refs, cand in zip(all_references, all_candidates)]
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
        
        return avg_loss, avg_bleu
    
    def train(self, train_loader, val_loader, epochs=10, clip=1.0, save_dir='models', save_freq=1, compute_bleu_freq=1):
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
            
            # 在验证集上评估 - 每N个epoch计算一次BLEU分数
            compute_bleu_score = (epoch + 1) % compute_bleu_freq == 0 or epoch == epochs - 1
            val_loss, val_bleu = self.evaluate(val_loader, compute_bleu_score)
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(val_bleu)
            
            # 记录学习率（如果使用了调度器）
            lr = self.scheduler.get_lr() if self.scheduler is not None else self.optimizer.param_groups[0]['lr']
            
            # 打印epoch结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            if compute_bleu_score:
                print(f"Val BLEU: {val_bleu:.4f}")
            print(f"Learning Rate: {lr:.6f}")
            
            # 保存表现最好的模型或按频率保存
            if val_loss < self.best_val_loss or (epoch + 1) % save_freq == 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_path = os.path.join(save_dir, 'best_model.pt')
                else:
                    save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
                
                # 减少保存的内容以加快保存速度
                save_dict = {
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }
                
                # 只在最佳模型保存优化器状态
                if val_loss < self.best_val_loss:
                    save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                    save_dict['train_losses'] = self.train_losses
                    save_dict['val_losses'] = self.val_losses
                    save_dict['val_bleu_scores'] = self.val_bleu_scores
                
                # 异步保存模型以减少阻塞
                torch.save(save_dict, save_path)
                print(f"Model saved to {save_path}")
        
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
    epochs = 5
    lr = 0.0001
    clip = 1.0
    
    # 性能优化参数
    use_fp16 = True and torch.cuda.is_available()  # 只有在CUDA可用时才启用混合精度训练
    grad_accumulation_steps = 1  # 梯度累积步数
    
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
    
    # 创建数据加载器 - 适配不同环境的配置
    # Windows系统下num_workers设为0以避免潜在问题
    import platform
    num_workers = 0 if platform.system() == 'Windows' else min(4, os.cpu_count() // 2)
    prefetch_factor = 2 if num_workers > 0 else None  # 仅在多进程时使用预取
    pin_memory = torch.cuda.is_available()  # 仅在CUDA可用时使用锁页内存
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory
    )
    
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
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params/1e6:.2f}M")
    
    # 创建训练器 - 启用混合精度和梯度累积
    trainer = Trainer(model, src_vocab, tgt_vocab, use_fp16=use_fp16, grad_accumulation_steps=grad_accumulation_steps)
    
    # 设置优化器和调度器
    trainer.setup_optimizer(lr=lr)
    trainer.setup_scheduler(d_model=d_model, warmup_steps=4000)
    
    # 开始训练
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=epochs, clip=clip, compute_bleu_freq=3)