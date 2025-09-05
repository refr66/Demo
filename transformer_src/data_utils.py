import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re
from datasets import load_dataset
import os
import pickle
import multiprocessing as mp
from functools import partial
import time
from typing import List, Dict, Tuple, Any

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
        # 使用多进程加速词频统计
        if len(corpus) > 1000 and mp.cpu_count() > 1:
            # 将语料库分成多个部分
            chunk_size = max(1000, len(corpus) // (mp.cpu_count() - 1))
            chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]
            
            # 创建进程池
            with mp.Pool(processes=min(mp.cpu_count() - 1, len(chunks))) as pool:
                # 每个进程处理一个语料块
                chunk_freqs = pool.map(self._count_tokens_in_chunk, chunks)
            
            # 合并所有进程的结果
            for chunk_freq in chunk_freqs:
                self.freq.update(chunk_freq)
        else:
            # 小规模语料库直接处理
            for text in corpus:
                tokens = self.tokenizer.tokenize(text)
                self.freq.update(tokens)
        
        # 添加满足最小频率要求的词到词汇表
        for token, freq in self.freq.items():
            if freq >= min_freq:
                self._add_token(token)
    
    def _count_tokens_in_chunk(self, chunk):
        """在进程池中使用的词频统计函数"""
        chunk_freq = Counter()
        for text in chunk:
            tokens = self.tokenizer.tokenize(text)
            chunk_freq.update(tokens)
        return chunk_freq
    
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
        # 优化的正则表达式模式，提高分词速度
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
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_len=100, use_cache=True):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.use_cache = use_cache
        # 预计算并缓存处理后的样本，减少重复计算
        self.cache = {} if use_cache else None
        
        # 对于大型数据集，使用分块预处理以减少内存使用
        self._preprocess_in_chunks()
    
    def _preprocess_in_chunks(self):
        """分块预处理数据以减少内存使用"""
        if not self.use_cache or len(self.src_data) < 1000:
            return  # 小型数据集不需要分块处理
        
        # 只预计算一小部分作为示例，其余按需计算
        sample_size = min(1000, len(self.src_data))
        for i in range(sample_size):
            self.__getitem__(i)  # 触发缓存
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        # 如果启用缓存且该索引已在缓存中，则直接返回缓存的结果
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
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
        
        result = {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_length': len(src_indices),
            'tgt_length': len(tgt_indices)
        }
        
        # 如果启用缓存，保存结果
        if self.use_cache:
            self.cache[idx] = result
        
        return result

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
    
    # 正确获取PAD_TOKEN的索引
    pad_idx = 0  # PAD_TOKEN的默认索引
    if hasattr(batch[0]['src'], 'vocab') and hasattr(batch[0]['src'].vocab, 'stoi'):
        pad_idx = batch[0]['src'].vocab.stoi.get(PAD_TOKEN, 0)
    elif hasattr(batch[0]['src'], 'stoi'):
        pad_idx = batch[0]['src'].stoi.get(PAD_TOKEN, 0)
    
    # 填充序列 - 优化版：使用向量化操作
    for i, (src_tensor, tgt_tensor) in enumerate(zip(src_tensors, tgt_tensors)):
        src_len = len(src_tensor)
        tgt_len = len(tgt_tensor)
        src_padded[i, :src_len] = src_tensor
        src_padded[i, src_len:] = pad_idx
        tgt_padded[i, :tgt_len] = tgt_tensor
        tgt_padded[i, tgt_len:] = pad_idx
    
    # 预计算掩码以在模型中使用，避免重复计算
    src_mask = (src_padded != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt_padded != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 创建序列长度的张量
    src_lengths = torch.tensor([item['src_length'] for item in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([item['tgt_length'] for item in batch], dtype=torch.long)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }

class OptimizedDataLoader(DataLoader):
    """优化的数据加载器，自动应用最佳实践"""
    def __init__(self, dataset, batch_size=32, shuffle=False, num_workers=None, pin_memory=True,
                 prefetch_factor=2, persistent_workers=True):
        # 根据CPU核心数和数据集大小自动确定num_workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # 限制最大worker数为8
            
        # 对于小数据集，减少worker数量
        if len(dataset) < 1000 and num_workers > 2:
            num_workers = 2
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

def load_translation_dataset(dataset_name='multi30k', split='train', src_lang='de', tgt_lang='en', cache_dir=None):
    """加载翻译数据集"""
    # 缓存文件名
    cache_filename = f'{dataset_name}_{split}_{src_lang}_{tgt_lang}.pkl'
    
    # 如果没有指定缓存目录，使用当前目录
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), 'data_cache')
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 完整的缓存文件路径
    cache_file = os.path.join(cache_dir, cache_filename)
    
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"Loading dataset from cache: {cache_file}")
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load dataset from cache: {e}")
    
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
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((src_data, tgt_data), f)
        except Exception as e:
            print(f"Failed to save dataset to cache: {e}")
        
        return src_data, tgt_data
    
    try:
        # 尝试加载数据集
        dataset = load_dataset(dataset_name, f'{src_lang}-{tgt_lang}', split=split, cache_dir=cache_dir)
        
        # 提取源语言和目标语言数据
        src_data = [example[src_lang] for example in dataset]
        tgt_data = [example[tgt_lang] for example in dataset]
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((src_data, tgt_data), f)
        except Exception as e:
            print(f"Failed to save dataset to cache: {e}")
        
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

def build_vocab_and_tokenizer(src_data, tgt_data, min_freq=2, cache_dir=None):
    """构建源语言和目标语言的词汇表和分词器"""
    # 缓存文件名
    cache_filename = f'vocab_{min_freq}.pkl'
    
    # 如果没有指定缓存目录，使用当前目录
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), 'data_cache')
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 完整的缓存文件路径
    cache_file = os.path.join(cache_dir, cache_filename)
    
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"Loading vocab from cache: {cache_file}")
                src_vocab, tgt_vocab = pickle.load(f)
                return src_vocab, tgt_vocab
        except Exception as e:
            print(f"Failed to load vocab from cache: {e}")
    
    # 创建分词器
    tokenizer = SimpleTokenizer()
    
    # 创建源语言词汇表
    src_vocab = Vocabulary(tokenizer)
    src_vocab.build_from_corpus(src_data, min_freq)
    
    # 创建目标语言词汇表
    tgt_vocab = Vocabulary(tokenizer)
    tgt_vocab.build_from_corpus(tgt_data, min_freq)
    
    # 保存到缓存
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
    except Exception as e:
        print(f"Failed to save vocab to cache: {e}")
    
    return src_vocab, tgt_vocab