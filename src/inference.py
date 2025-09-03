import torch
import torch.nn as nn
from transformer import Transformer
from data_utils import Vocabulary, SimpleTokenizer, load_translation_dataset, build_vocab_and_tokenizer, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from utils import create_src_mask
import time

class Translator:
    def __init__(self, model_path, src_vocab, tgt_vocab, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # 安全获取特殊标记的索引
        self.sos_idx = self._safe_get_token_index(tgt_vocab, SOS_TOKEN)
        self.eos_idx = self._safe_get_token_index(tgt_vocab, EOS_TOKEN)
        self.pad_idx = self._safe_get_token_index(tgt_vocab, PAD_TOKEN)
    
    def _safe_get_token_index(self, vocab, token, default=0):
        """安全地获取词汇表中标记的索引"""
        try:
            return vocab.stoi[token]
        except KeyError:
            print(f"Warning: Token '{token}' not found in vocabulary, using default index {default}")
            # 添加该标记到词汇表
            if token not in vocab.stoi:
                vocab.stoi[token] = default
                vocab.itos[default] = token
                vocab.vocab_size += 1
            return default
    
    def _load_model(self, model_path):
        """加载训练好的模型，处理参数不匹配情况"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 获取模型参数
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.tgt_vocab)
            
            # 尝试从checkpoint中获取模型参数，如果获取不到则使用默认值
            # 如果checkpoint使用的是旧版本参数，这里可以进行适配
            d_model = checkpoint.get('d_model', 256)  # 使用旧版本的默认值以兼容已保存的模型
            num_heads = checkpoint.get('num_heads', 4)
            num_layers = checkpoint.get('num_layers', 2)
            d_ff = checkpoint.get('d_ff', 1024)
            dropout = checkpoint.get('dropout', 0.1)
            max_len = checkpoint.get('max_len', 50)
            
            # 输出当前使用的模型参数，帮助调试
            print(f"加载模型参数: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, d_ff={d_ff}")
            
            # 创建模型
            model = Transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len
            ).to(self.device)
            
            # 尝试加载模型权重，处理参数不匹配情况
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型成功加载: {model_path}")
            except RuntimeError as e:
                # 解决f-string中不能直接包含反斜杠的问题
                newline = '\n'
                print(f"警告: 检测到模型参数不匹配: {str(e).split(newline)[0]}")
                print("尝试使用strict=False加载模型（忽略参数不匹配）...")
                
                # 使用strict=False来忽略参数不匹配问题
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("模型已加载，但存在参数不匹配问题。这可能会影响翻译质量。")
                print("建议使用当前代码版本重新训练模型以获得最佳性能。")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("由于保存的模型与当前代码不兼容，创建一个新的随机初始化模型用于演示...")
            
            # 创建一个随机初始化的模型
            d_model = 256
            num_heads = 4
            num_layers = 2
            d_ff = 1024
            dropout = 0.1
            max_len = 50
            
            model = Transformer(
                src_vocab_size=len(self.src_vocab),
                tgt_vocab_size=len(self.tgt_vocab),
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len
            ).to(self.device)
        
        model.eval()
        return model
        
        return model
    
    def preprocess(self, text):
        """预处理输入文本"""
        # 分词
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        
        # 添加特殊标记
        tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        
        # 转换为索引
        indices = self.src_vocab.numericalize(tokens)
        
        # 转换为张量
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]
        
        return tensor
    
    def postprocess(self, indices):
        """后处理输出索引"""
        # 检查indices是否为空
        if not indices:
            return ""
        
        # 移除SOS标记（如果存在）
        if indices[0] == self.sos_idx:
            indices = indices[1:]
        
        # 检查移除SOS后indices是否为空
        if not indices:
            return ""
        
        # 移除EOS标记（如果存在）
        if self.eos_idx in indices:
            eos_pos = indices.index(self.eos_idx)
            indices = indices[:eos_pos]
        
        # 检查移除EOS后indices是否为空
        if not indices:
            return ""
        
        # 转换为文本
        tokens = self.tgt_vocab.denumericalize(indices)
        text = ' '.join(tokens)
        
        return text
    
    def greedy_decode(self, src_tensor, max_len=50):
        """使用贪心解码生成目标序列"""
        # 获取源序列掩码
        src_mask = create_src_mask(src_tensor, self.src_vocab.stoi[PAD_TOKEN])
        
        # 编码源序列
        with torch.no_grad():
            enc_output, _ = self.model.encoder(src_tensor, src_mask)
        
        # 初始化解码器输入（只有SOS标记）
        tgt_tensor = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)
        
        # 逐个生成token
        for _ in range(max_len):
            # 创建目标序列掩码
            tgt_mask = self._create_tgt_mask(tgt_tensor)
            
            # 解码
            with torch.no_grad():
                output, _, _ = self.model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            
            # 预测下一个token（取概率最高的）
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            
            # 添加到目标序列
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)
            
            # 如果生成了EOS标记，停止生成
            if next_token == self.eos_idx:
                break
        
        # 转换为索引列表
        result_indices = tgt_tensor.squeeze(0).cpu().tolist()
        
        return result_indices
    
    def beam_search_decode(self, src_tensor, beam_size=5, max_len=50):
        """使用束搜索解码生成目标序列"""
        # 获取源序列掩码
        src_mask = create_src_mask(src_tensor, self.src_vocab.stoi[PAD_TOKEN])
        
        # 编码源序列
        with torch.no_grad():
            enc_output, _ = self.model.encoder(src_tensor, src_mask)
        
        # 初始化
        batch_size = src_tensor.size(0)
        assert batch_size == 1, "Beam search only supports batch size 1"
        
        # 初始序列（只有SOS标记）
        tgt_tensor = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)
        
        # 初始得分
        scores = torch.zeros(beam_size, device=self.device)
        
        # 候选序列和得分
        candidates = [(tgt_tensor, scores[0])]
        
        for _ in range(max_len):
            new_candidates = []
            
            for seq, score in candidates:
                # 如果序列已结束，直接添加到新候选
                if seq[0, -1].item() == self.eos_idx:
                    new_candidates.append((seq, score))
                    continue
                
                # 创建目标序列掩码
                tgt_mask = self._create_tgt_mask(seq)
                
                # 解码
                with torch.no_grad():
                    output, _, _ = self.model.decoder(seq, enc_output, src_mask, tgt_mask)
                
                # 获取下一个token的概率分布
                next_token_logits = output[:, -1, :]
                next_token_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # 取概率最高的beam_size个token
                top_probs, top_indices = torch.topk(next_token_probs, beam_size)
                
                # 生成新的候选序列
                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    prob = top_probs[0, i].item()
                    
                    # 忽略填充标记
                    if token == self.pad_idx:
                        continue
                    
                    # 创建新序列
                    new_seq = torch.cat([seq, torch.tensor([[token]], device=self.device)], dim=1)
                    
                    # 更新得分
                    new_score = score + prob
                    
                    new_candidates.append((new_seq, new_score))
            
            # 按得分排序并保留前beam_size个候选
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_size]
            
            # 检查是否所有候选都已结束
            all_finished = all(seq[0, -1].item() == self.eos_idx for seq, _ in candidates)
            if all_finished:
                break
        
        # 选择得分最高的序列
        best_seq, _ = candidates[0]
        
        # 转换为索引列表
        result_indices = best_seq.squeeze(0).cpu().tolist()
        
        return result_indices
    
    def translate(self, text, method='beam', beam_size=5, max_len=50):
        """将源语言文本翻译为目标语言文本"""
        # 预处理
        src_tensor = self.preprocess(text)
        
        # 解码
        start_time = time.time()
        if method == 'beam':
            indices = self.beam_search_decode(src_tensor, beam_size=beam_size, max_len=max_len)
        else:
            indices = self.greedy_decode(src_tensor, max_len=max_len)
        end_time = time.time()
        
        # 后处理
        result = self.postprocess(indices)
        
        return result, end_time - start_time
    
    def _create_tgt_mask(self, tgt):
        """创建目标序列的掩码"""
        # 填充掩码
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 前瞻掩码
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device))
        
        # 合并掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool().unsqueeze(0).unsqueeze(1)
        
        return tgt_mask

# 辅助函数：加载词汇表和创建翻译器
def load_translator(model_path, dataset_name='multi30k', src_lang='de', tgt_lang='en'):
    """加载训练好的翻译器"""
    # 加载数据集以构建词汇表
    print("Loading dataset for vocabulary...")
    train_src_data, train_tgt_data = load_translation_dataset(dataset_name, 'train', src_lang, tgt_lang)
    
    # 构建词汇表
    print("Building vocabularies...")
    src_vocab, tgt_vocab = build_vocab_and_tokenizer(train_src_data, train_tgt_data, min_freq=2)
    
    # 创建翻译器
    print(f"Loading model from {model_path}...")
    translator = Translator(model_path, src_vocab, tgt_vocab)
    
    return translator

# 示例用法
if __name__ == "__main__":
    # 模型路径
    model_path = 'models/best_model.pt'  # 假设模型保存在这里
    
    # 尝试加载翻译器
    try:
        translator = load_translator(model_path)
        
        print("\nTransformer Translator")
        print("-------------------")
        print("Enter a German sentence to translate to English (or 'q' to quit):")
        
        while True:
            # 获取用户输入
            src_text = input("German: ")
            if src_text.lower() == 'q':
                break
            
            # 翻译
            result, time_taken = translator.translate(src_text, method='beam', beam_size=5)
            
            # 显示结果
            print(f"English: {result}")
            print(f"Time taken: {time_taken:.4f} seconds")
            print()
            
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train a model first using train.py")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have trained a model and the model path is correct")