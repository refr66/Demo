import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re
from datasets import load_dataset
import os

# 特殊标记定义
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

class Vocabulary:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else SimpleTokenizer()
        self.itos = {}
        self.stoi = {}
        self.freq = Counter()
        self.vocab_size = 0
        
        # 添加特殊标记
        self.add_special_tokens()
    
    def add_special_tokens(self):
        """添加特殊标记到词汇表"""
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for token in special_tokens:
            self._add_token(token)
    
    def _add_token(self, token):
        """将单个标记添加到词汇表"""
        if token not in self.stoi:
            self.stoi[token] = self.vocab_size
            self.itos[self.vocab_size] = token
            self.vocab_size += 1
    
    def build_from_corpus(self, corpus, min_freq=1):
        """从语料库构建词汇表"""
        # 统计词频
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            self.freq.update(tokens)
        
        # 添加满足最小频率要求的词到词汇表
        for token, freq in self.freq.items():
            if freq >= min_freq:
                self._add_token(token)
    
    def numericalize(self, tokens):
        """将标记列表转换为索引列表"""
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]
    
    def denumericalize(self, indices):
        """将索引列表转换为标记列表"""
        return [self.itos.get(idx, UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return self.vocab_size

class SimpleTokenizer:
    def __init__(self):
        # 简单的分词规则，可以根据需要替换为更复杂的分词器
        self.pattern = re.compile(r'([,.!?"()])')
    
    def tokenize(self, text):
        """将文本分割成标记列表"""
        # 转换为小写
        text = text.lower()
        # 分割标点符号
        text = self.pattern.sub(r' \1 ', text)
        # 分割空白字符
        tokens = text.split()
        return tokens

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_len=100):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        # 获取源语言和目标语言文本
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # 分词
        src_tokens = self.src_vocab.tokenizer.tokenize(src_text)
        tgt_tokens = self.tgt_vocab.tokenizer.tokenize(tgt_text)
        
        # 添加特殊标记并截断过长的序列
        src_tokens = [SOS_TOKEN] + src_tokens[:self.max_len-2] + [EOS_TOKEN]
        tgt_tokens = [SOS_TOKEN] + tgt_tokens[:self.max_len-2] + [EOS_TOKEN]
        
        # 转换为索引
        src_indices = self.src_vocab.numericalize(src_tokens)
        tgt_indices = self.tgt_vocab.numericalize(tgt_tokens)
        
        # 转换为张量
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_length': len(src_indices),
            'tgt_length': len(tgt_indices)
        }

def collate_fn(batch):
    """用于DataLoader的collate函数，处理变长序列"""
    # 获取批次中最长的源序列和目标序列长度
    max_src_len = max(item['src_length'] for item in batch)
    max_tgt_len = max(item['tgt_length'] for item in batch)
    
    # 提取源序列和目标序列
    src_tensors = [item['src'] for item in batch]
    tgt_tensors = [item['tgt'] for item in batch]
    
    # 创建填充后的张量
    batch_size = len(batch)
    src_padded = torch.zeros((batch_size, max_src_len), dtype=torch.long)
    tgt_padded = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
    
    # 填充序列
    pad_idx = batch[0]['src'].vocab.stoi[PAD_TOKEN] if hasattr(batch[0]['src'], 'vocab') else 0
    
    for i, (src_tensor, tgt_tensor) in enumerate(zip(src_tensors, tgt_tensors)):
        src_len = len(src_tensor)
        tgt_len = len(tgt_tensor)
        src_padded[i, :src_len] = src_tensor
        src_padded[i, src_len:] = pad_idx
        tgt_padded[i, :tgt_len] = tgt_tensor
        tgt_padded[i, tgt_len:] = pad_idx
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': torch.tensor([item['src_length'] for item in batch]),
        'tgt_lengths': torch.tensor([item['tgt_length'] for item in batch])
    }

def load_translation_dataset(dataset_name='multi30k', split='train', src_lang='de', tgt_lang='en', cache_dir=None):
    """加载翻译数据集"""
    # 首先尝试从本地data文件夹加载数据集
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # 根据split参数确定文件名
    if split == 'train':
        src_file = os.path.join(base_path, f'train.{src_lang}.gz')
        tgt_file = os.path.join(base_path, f'train.{tgt_lang}.gz')
    elif split == 'validation' or split == 'val':
        src_file = os.path.join(base_path, f'val.{src_lang}.gz')
        tgt_file = os.path.join(base_path, f'val.{tgt_lang}.gz')
    else:
        # 对于其他分割，仍然尝试从Hugging Face Hub加载
        src_file = None
        tgt_file = None
    
    # 检查本地文件是否存在
    if src_file and tgt_file and os.path.exists(src_file) and os.path.exists(tgt_file):
        print(f"Loading dataset from local files: {src_file} and {tgt_file}")
        
        # 读取压缩文件
        src_data = []
        tgt_data = []
        
        import gzip
        
        # 读取源语言数据
        with gzip.open(src_file, 'rt', encoding='utf-8') as f:
            for line in f:
                src_data.append(line.strip())
        
        # 读取目标语言数据
        with gzip.open(tgt_file, 'rt', encoding='utf-8') as f:
            for line in f:
                tgt_data.append(line.strip())
        
        return src_data, tgt_data
    
    # 如果没有指定缓存目录，使用当前目录
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), 'data_cache')
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 尝试加载数据集
        dataset = load_dataset(dataset_name, f'{src_lang}-{tgt_lang}', split=split, cache_dir=cache_dir)
        
        # 提取源语言和目标语言数据
        src_data = [example[src_lang] for example in dataset]
        tgt_data = [example[tgt_lang] for example in dataset]
        
        return src_data, tgt_data
    except Exception as e:
        print(f"无法加载数据集 '{dataset_name}'，错误: {e}")
        print("使用示例数据集进行演示...")
        
        # 提供简单的示例数据集（德语到英语）
        if src_lang == 'de' and tgt_lang == 'en':
            if split == 'train':
                # 简单的训练示例数据
                de_examples = [
                    'ein mann fährt ein motorrad.',
                    'eine frau liest ein buch.',
                    'die katze sitzt auf der couch.',
                    'wir gehen in den park.',
                    'ich esse ein apfel.'
                ] * 100  # 复制以增加数据集大小
                en_examples = [
                    'a man is riding a motorcycle.',
                    'a woman is reading a book.',
                    'the cat is sitting on the couch.',
                    'we are going to the park.',
                    'i am eating an apple.'
                ] * 100
            else:
                # 验证/测试示例数据
                de_examples = [
                    'der Hund bellt laut.',
                    'wir trinken kaffee.',
                    'das kind spielt mit einem ball.'
                ]
                en_examples = [
                    'the dog is barking loudly.',
                    'we are drinking coffee.',
                    'the child is playing with a ball.'
                ]
            return de_examples, en_examples
        else:
            # 对于其他语言对，返回空数据集
            return [], []

def build_vocab_and_tokenizer(src_data, tgt_data, min_freq=2):
    """构建源语言和目标语言的词汇表和分词器"""
    # 创建分词器
    tokenizer = SimpleTokenizer()
    
    # 创建源语言词汇表
    src_vocab = Vocabulary(tokenizer)
    src_vocab.build_from_corpus(src_data, min_freq)
    
    # 创建目标语言词汇表
    tgt_vocab = Vocabulary(tokenizer)
    tgt_vocab.build_from_corpus(tgt_data, min_freq)
    
    return src_vocab, tgt_vocab