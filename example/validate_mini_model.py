import torch
import os
import sys

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加src目录到Python路径
sys.path.append(os.path.join(current_dir, 'src'))

# 导入需要的模块
try:
    from transformer import Transformer
    from data_utils import Vocabulary, SimpleTokenizer
except ImportError:
    # 如果第一种导入方式失败，尝试直接导入（假设我们已经在正确的工作目录）
    try:
        from src.transformer import Transformer
        from src.data_utils import Vocabulary, SimpleTokenizer
    except ImportError:
        print("错误: 无法导入必要的模块")
        print("请确保您的工作目录是项目根目录，并且src目录包含所需的文件")
        sys.exit(1)

class ModelValidator:
    def __init__(self, model_path):
        """初始化验证器，加载模型"""
        self.model_path = model_path
        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载已保存的模型"""
        print(f"正在加载模型: {self.model_path}")
        
        try:
            # 加载保存的模型数据（设置weights_only=False以加载自定义类）
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 检查checkpoint中是否包含所有必要的键
            required_keys = ['src_vocab', 'tgt_vocab', 'd_model', 'num_heads', 
                            'num_layers', 'd_ff', 'dropout', 'max_len', 'model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"模型文件中缺少必要的键: {key}")
            
            # 提取词汇表
            self.src_vocab = checkpoint['src_vocab']
            self.tgt_vocab = checkpoint['tgt_vocab']
            
            # 提取模型参数
            d_model = checkpoint['d_model']
            num_heads = checkpoint['num_heads']
            num_layers = checkpoint['num_layers']
            d_ff = checkpoint['d_ff']
            dropout = checkpoint['dropout']
            max_len = checkpoint['max_len']
            
            # 创建模型实例
            self.model = Transformer(
                src_vocab_size=len(self.src_vocab),
                tgt_vocab_size=len(self.tgt_vocab),
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len
            )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            print(f"模型加载成功！")
            print(f"源语言词汇表大小: {len(self.src_vocab)}")
            print(f"目标语言词汇表大小: {len(self.tgt_vocab)}")
            print(f"模型参数: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            print("请确保模型文件格式正确且完整")
            sys.exit(1)
    
    def create_tgt_mask(self, tgt):
        """创建目标序列的掩码"""
        # 获取特殊标记的索引
        pad_idx = self.src_vocab.stoi['<PAD>']
        
        # 填充掩码
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 前瞻掩码
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device))
        
        # 合并掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask.bool().unsqueeze(0).unsqueeze(1)
        
        return tgt_mask
    
    def greedy_decode(self, src, max_len=50):
        """使用贪婪搜索解码"""
        src = src.to(self.device)
        batch_size = src.size(0)
        
        # 获取特殊标记的索引
        pad_idx = self.src_vocab.stoi['<PAD>']
        sos_idx = self.src_vocab.stoi['<SOS>']
        eos_idx = self.src_vocab.stoi['<EOS>']
        
        # 初始化输出序列，只包含SOS标记
        tgt = torch.full((batch_size, 1), sos_idx, device=self.device, dtype=torch.long)
        
        # 源序列掩码
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 自回归解码
        for _ in range(max_len - 1):
            # 目标序列掩码
            tgt_mask = self.create_tgt_mask(tgt)
            
            # 前向传播
            with torch.no_grad():
                output, _, _, _ = self.model(src, tgt, src_mask, tgt_mask)
            
            # 取最后一个位置的输出
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # 将预测的标记添加到输出序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果预测到EOS标记，停止解码
            if (next_token == eos_idx).all():
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
        tgt_tensor = self.greedy_decode(src_tensor)
        
        # 转换为文本
        tgt_indices = tgt_tensor.squeeze(0).cpu().tolist()
        
        # 移除SOS和EOS标记
        if tgt_indices[0] == self.src_vocab.stoi['<SOS>']:
            tgt_indices = tgt_indices[1:]
        if self.src_vocab.stoi['<EOS>'] in tgt_indices:
            eos_pos = tgt_indices.index(self.src_vocab.stoi['<EOS>'])
            tgt_indices = tgt_indices[:eos_pos]
        
        # 转换为文本
        tgt_tokens = self.tgt_vocab.denumericalize(tgt_indices)
        
        return ' '.join(tgt_tokens)
    
    def validate_with_sentences(self, sentences):
        """使用一组句子验证模型"""
        print("\n开始验证模型翻译能力...")
        
        for sentence in sentences:
            try:
                translation = self.translate(sentence)
                print(f"德语: {sentence}")
                print(f"英语: {translation}")
                print()
            except Exception as e:
                print(f"翻译 '{sentence}' 时出错: {str(e)}")
                print()

def main():
    # 模型文件路径
    model_path = "best_mini_model.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 '{model_path}'")
        print("请先运行 mini-data/mini_train.py 来训练模型")
        sys.exit(1)
    
    print("===== Transformer模型验证工具 ======")
    print("此工具用于验证已训练保存的Transformer翻译模型")
    print(f"当前验证的模型: {model_path}")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("====================================")
    
    # 创建验证器实例
    validator = ModelValidator(model_path)
    
    # 测试句子
    test_sentences = [
        "ich bin ein student.",
        "die katze schläft.",
        "ich trinke wasser.",
        "ich liebe dich.",
        "das ist ein buch."
    ]
    
    # 使用测试句子验证模型
    validator.validate_with_sentences(test_sentences)
    
    # 交互式验证
    print("\n交互式翻译测试 (输入 'exit' 退出):")
    while True:
        try:
            user_input = input("请输入德语句子: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            translation = validator.translate(user_input)
            print(f"英语翻译: {translation}")
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"翻译时出错: {str(e)}")
            print()
    
    print("验证完成！")

# 主函数入口
if __name__ == "__main__":
    main()
    
    # 创建验证器实例
    validator = ModelValidator(model_path)
    
    # 测试句子
    test_sentences = [
        "ich bin ein student.",
        "die katze schläft.",
        "ich trinke wasser.",
        "ich liebe dich.",
        "das ist ein buch."
    ]
    
    # 使用测试句子验证模型
    validator.validate_with_sentences(test_sentences)
    
    # 交互式验证
    print("\n交互式翻译测试 (输入 'exit' 退出):")
    while True:
        try:
            user_input = input("请输入德语句子: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            translation = validator.translate(user_input)
            print(f"英语翻译: {translation}")
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"翻译时出错: {str(e)}")
            print()
    
    print("验证完成！")