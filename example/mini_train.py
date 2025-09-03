import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import sys

# 获取当前目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加src目录到Python路径，这样就能正确导入src中的模块
sys.path.append(os.path.join(parent_dir, 'src'))

# 导入需要的模块
from transformer import Transformer
from data_utils import Vocabulary, SimpleTokenizer, TranslationDataset, collate_fn
from utils import NoamOpt

class MiniTrainer:
    def __init__(self, model, src_vocab, tgt_vocab, model_params=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.model_params = model_params
        
        # 获取特殊标记的索引
        self.pad_idx = src_vocab.stoi['<PAD>']
        self.sos_idx = src_vocab.stoi['<SOS>']
        self.eos_idx = src_vocab.stoi['<EOS>']
        
        # 定义损失函数，忽略填充标记
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        # 初始化优化器和调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98))
        
    def create_tgt_mask(self, tgt):
        """创建目标序列的掩码"""
        # 填充掩码
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 前瞻掩码
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device))
        
        # 合并掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool().unsqueeze(0).unsqueeze(1)
        
        return tgt_mask
        
    def train_epoch(self, train_loader, clip=1.0):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
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
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
        
    def evaluate(self, val_loader):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
        
    def train(self, train_loader, val_loader, epochs=5):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"Training loss: {train_loss:.4f}")
            
            # 验证
            val_loss = self.evaluate(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 模型保存到mini-data目录下
                self.save_model(os.path.join(current_dir, "best_mini_model.pt"))
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
    def save_model(self, model_path):
        """保存模型"""
        # 使用传入的model_params保存模型参数，避免直接访问model的属性
        if self.model_params is None:
            # 如果没有传入model_params，则使用占位符值
            self.model_params = {
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'd_ff': 512,
                'dropout': 0.1,
                'max_len': 50
            }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'd_model': self.model_params.get('d_model'),
            'num_heads': self.model_params.get('num_heads'),
            'num_layers': self.model_params.get('num_layers'),
            'd_ff': self.model_params.get('d_ff'),
            'dropout': self.model_params.get('dropout'),
            'max_len': self.model_params.get('max_len')
        }, model_path)
        # 模型保存位置为mini-data目录下

    def greedy_decode(self, src, max_len=50):
        """使用贪婪搜索解码"""
        src = src.to(self.device)
        batch_size = src.size(0)
        
        # 初始化输出序列，只包含SOS标记
        tgt = torch.full((batch_size, 1), self.sos_idx, device=self.device, dtype=torch.long)
        
        # 源序列掩码
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 自回归解码
        for _ in range(max_len - 1):
            # 目标序列掩码
            tgt_mask = self.create_tgt_mask(tgt)
            
            # 前向传播
            output, _, _, _ = self.model(src, tgt, src_mask, tgt_mask)
            
            # 取最后一个位置的输出
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # 将预测的标记添加到输出序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果预测到EOS标记，停止解码
            if (next_token == self.eos_idx).all():
                break
        
        return tgt

    def translate(self, text, tokenizer=None):
        """翻译单个句子"""
        if tokenizer is None:
            tokenizer = SimpleTokenizer()
            
        # 分词
        tokens = tokenizer.tokenize(text.lower())
        
        # 添加特殊标记
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        
        # 转换为索引
        indices = self.src_vocab.numericalize(tokens)
        
        # 转换为张量
        src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # 解码
        with torch.no_grad():
            tgt_tensor = self.greedy_decode(src_tensor)
        
        # 转换为文本
        tgt_indices = tgt_tensor.squeeze(0).cpu().tolist()
        
        # 移除SOS和EOS标记
        if tgt_indices[0] == self.sos_idx:
            tgt_indices = tgt_indices[1:]
        if self.eos_idx in tgt_indices:
            eos_pos = tgt_indices.index(self.eos_idx)
            tgt_indices = tgt_indices[:eos_pos]
        
        # 转换为文本
        tgt_tokens = self.tgt_vocab.denumericalize(tgt_indices)
        
        return ' '.join(tgt_tokens)

# 读取小数据集
def load_mini_dataset(file_path):
    """加载小型数据集文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

# 主函数
if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载小数据集
    print("Loading mini dataset...")
    train_src_data = load_mini_dataset(os.path.join(current_dir, 'mini_train.de'))
    train_tgt_data = load_mini_dataset(os.path.join(current_dir, 'mini_train.en'))
    val_src_data = load_mini_dataset(os.path.join(current_dir, 'mini_val.de'))
    val_tgt_data = load_mini_dataset(os.path.join(current_dir, 'mini_val.en'))
    
    print(f"Train data size: {len(train_src_data)} sentence pairs")
    print(f"Validation data size: {len(val_src_data)} sentence pairs")
    
    # 构建词汇表
    print("Building vocabularies...")
    tokenizer = SimpleTokenizer()
    
    # 创建源语言词汇表
    src_vocab = Vocabulary(tokenizer)
    src_vocab.build_from_corpus(train_src_data, min_freq=1)
    
    # 创建目标语言词汇表
    tgt_vocab = Vocabulary(tokenizer)
    tgt_vocab.build_from_corpus(train_tgt_data, min_freq=1)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # 模型参数（使用较小的模型因为数据集很小）
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    dropout = 0.1
    max_len = 50
    
    # 训练参数
    batch_size = 2
    epochs = 30  # 增加训练轮数以提升模型准确性
    
    # 创建数据集
    train_dataset = TranslationDataset(train_src_data, train_tgt_data, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_src_data, val_tgt_data, src_vocab, tgt_vocab, max_len)
    
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print("Creating mini Transformer model...")
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
    
    # 创建训练器，传入模型参数
    model_params = {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'max_len': max_len
    }
    trainer = MiniTrainer(model, src_vocab, tgt_vocab, model_params)
    
    # 开始训练
    print("Starting training with mini dataset...")
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 测试翻译
    print("\nTesting translation with trained model:")
    test_sentences = [
        "ich bin ein student.",
        "die katze schläft.",
        "ich trinke wasser."
    ]
    
    for sentence in test_sentences:
        translation = trainer.translate(sentence)
        print(f"German: {sentence}")
        print(f"English: {translation}")
        print()
    
    print("Training completed! You can now use the trained model for translation.")
    print("Model saved as: best_mini_model.pt")