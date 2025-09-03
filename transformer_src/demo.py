import os
import sys

# 检查必要的依赖包
missing_packages = []
try:
    import torch
except ImportError:
    missing_packages.append('torch')

try:
    import numpy
except ImportError:
    missing_packages.append('numpy')

try:
    import matplotlib
except ImportError:
    missing_packages.append('matplotlib')

try:
    import nltk
except ImportError:
    missing_packages.append('nltk')

try:
    import datasets
except ImportError:
    missing_packages.append('datasets')

# 如果缺少依赖包，显示友好的错误提示
if missing_packages:
    print("错误：缺少以下必要的Python包：")
    for pkg in missing_packages:
        print(f"- {pkg}")
    print()
    print("请使用pip安装这些依赖：")
    print(f"pip install {' '.join(missing_packages)}")
    print()
    print("或者安装所有依赖：")
    print("pip install -r transformer_implementation/requirements.txt")
    print()
    print("对于PyTorch，您可能需要根据您的CUDA版本安装特定版本：")
    print("访问 https://pytorch.org/get-started/locally/ 获取详细安装指南")
    sys.exit(1)

# 导入项目模块
import argparse
from transformer import Transformer
from data_utils import load_translation_dataset, build_vocab_and_tokenizer, TranslationDataset, collate_fn
from train import Trainer
from inference import Translator, load_translator
from utils import plot_training_progress, save_vocabulary

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Transformer Demo')
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'infer', 'demo'],
                        help='运行模式: train(训练模型), infer(使用模型翻译), demo(完整流程演示)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                        help='预训练模型路径')
    parser.add_argument('--vocab_dir', type=str, default='vocab',
                        help='词汇表保存目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--num_layers', type=int, default=2, help='编码器和解码器层数')
    parser.add_argument('--d_ff', type=int, default=1024, help='前馈网络内部维度')
    parser.add_argument('--max_len', type=int, default=50, help='最大序列长度')
    parser.add_argument('--dataset', type=str, default='multi30k', help='数据集名称')
    parser.add_argument('--src_lang', type=str, default='de', help='源语言')
    parser.add_argument('--tgt_lang', type=str, default='en', help='目标语言')
    return parser.parse_args()

def prepare_data(args):
    """准备数据和词汇表"""
    print("加载数据集...")
    # 加载训练数据、验证数据和测试数据
    train_src_data, train_tgt_data = load_translation_dataset(args.dataset, 'train', args.src_lang, args.tgt_lang)
    val_src_data, val_tgt_data = load_translation_dataset(args.dataset, 'validation', args.src_lang, args.tgt_lang)
    test_src_data, test_tgt_data = load_translation_dataset(args.dataset, 'test', args.src_lang, args.tgt_lang)
    
    # 检查词汇表是否已存在
    src_vocab_path = os.path.join(args.vocab_dir, f'{args.src_lang}_vocab.txt')
    tgt_vocab_path = os.path.join(args.vocab_dir, f'{args.tgt_lang}_vocab.txt')
    
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print("加载已存在的词汇表...")
        from utils import load_vocabulary
        src_vocab = load_vocabulary(src_vocab_path)
        tgt_vocab = load_vocabulary(tgt_vocab_path)
    else:
        print("构建新的词汇表...")
        # 构建词汇表
        src_vocab, tgt_vocab = build_vocab_and_tokenizer(train_src_data, train_tgt_data, min_freq=2)
        
        # 保存词汇表
        print("保存词汇表...")
        os.makedirs(args.vocab_dir, exist_ok=True)
        save_vocabulary(src_vocab, src_vocab_path)
        save_vocabulary(tgt_vocab, tgt_vocab_path)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = TranslationDataset(train_src_data, train_tgt_data, src_vocab, tgt_vocab, args.max_len)
    val_dataset = TranslationDataset(val_src_data, val_tgt_data, src_vocab, tgt_vocab, args.max_len)
    test_dataset = TranslationDataset(test_src_data, test_tgt_data, src_vocab, tgt_vocab, args.max_len)
    
    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab

def train_model(args, train_dataset, val_dataset, src_vocab, tgt_vocab):
    """训练模型"""
    import torch.utils.data as data
    
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=0.1,
        max_len=args.max_len
    )
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    trainer = Trainer(model, src_vocab, tgt_vocab, device)
    
    # 设置优化器和调度器
    trainer.setup_optimizer(lr=args.lr)
    trainer.setup_scheduler(d_model=args.d_model, warmup_steps=400)
    
    # 开始训练
    print("开始训练...")
    trainer.train(train_loader, val_loader, epochs=args.epochs, clip=1.0, save_dir=args.model_dir)
    
    # 可视化训练过程
    plot_training_progress(
        trainer.train_losses,
        trainer.val_losses,
        trainer.val_bleu_scores,
        save_path=os.path.join(args.model_dir, 'training_progress.png')
    )
    
    return trainer.best_val_loss

def translate_text(args):
    """使用训练好的模型进行翻译"""
    try:
        # 加载翻译器
        translator = load_translator(args.model_path, args.dataset, args.src_lang, args.tgt_lang)
        
        print(f"\nTransformer 翻译演示 ({args.src_lang} → {args.tgt_lang})")
        print("-------------------------")
        print("输入句子进行翻译（输入'q'退出）：")
        
        while True:
            # 获取用户输入
            src_text = input(f"{args.src_lang}: ")
            if src_text.lower() == 'q':
                break
            
            # 翻译
            result, time_taken = translator.translate(src_text, method='beam', beam_size=5)
            
            # 显示结果
            print(f"{args.tgt_lang}: {result}")
            print(f"翻译用时: {time_taken:.4f} 秒")
            print()
    except FileNotFoundError:
        print(f"模型文件未找到: {args.model_path}")
        print("请先使用 --mode train 训练模型")
    except Exception as e:
        print(f"错误: {e}")

def run_demo(args):
    """运行完整的演示流程"""
    print("=== Transformer 完整演示 ===")
    
    # 准备数据
    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = prepare_data(args)
    
    # 训练模型（使用少量轮数进行演示）
    print("\n=== 开始训练模型（演示模式）===")
    best_val_loss = train_model(args, train_dataset, val_dataset, src_vocab, tgt_vocab)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 使用模型进行翻译
    print("\n=== 使用训练好的模型进行翻译 ===")
    translate_text(args)
    
    print("\n=== 演示完成 ===")
    print("提示：")
    print("1. 要进行完整训练，请使用命令：python demo.py --mode train --epochs 20")
    print("2. 要仅使用训练好的模型翻译，请使用命令：python demo.py --mode infer")
    print("3. 可以通过修改参数来调整模型大小和训练设置")

def main():
    """主函数"""
    args = parse_args()
    
    # 根据模式运行相应的功能
    if args.mode == 'train':
        # 准备数据
        train_dataset, val_dataset, _, src_vocab, tgt_vocab = prepare_data(args)
        # 训练模型
        train_model(args, train_dataset, val_dataset, src_vocab, tgt_vocab)
    elif args.mode == 'infer':
        # 翻译文本
        translate_text(args)
    else:
        # 运行完整演示
        run_demo(args)

if __name__ == "__main__":
    main()