import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# 导入并行策略实现
from .data_parallel import DataParallel, train_parallel, create_distributed_loader
from .pipeline_parallel import PipelineParallel, create_pipeline_model, train_pipeline
from .tensor_parallel import TensorParallel, create_tensor_parallel_model, train_tensor_parallel, ParallelLinear

# 定义一个简单的数据集用于演示
class SimpleDataset(Dataset):
    """
    简单数据集，用于演示并行训练
    """
    def __init__(self, num_samples=1000, input_dim=10, output_dim=5):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 随机生成数据
        self.X = torch.randn(num_samples, input_dim)
        # 创建一个简单的线性关系作为标签
        weights = torch.randn(input_dim, output_dim)
        bias = torch.randn(output_dim)
        self.y = torch.matmul(self.X, weights) + bias
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义一个简单的模型
class SimpleModel(nn.Module):
    """
    简单的多层感知器模型，用于演示并行训练
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=5, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 定义一个更深的模型，用于流水线并行演示
class DeepModel(nn.Module):
    """
    较深的模型，用于演示流水线并行
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=5, num_layers=8):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 示例1: 使用数据并行训练模型
def example_data_parallel():
    """
    数据并行示例
    """
    print("\n===== 数据并行示例 =====")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(num_samples=10000, input_dim=10, output_dim=5)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 创建模型、损失函数和优化器
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 获取可用设备
    device_ids = list(range(min(2, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"使用设备: {device_ids}")
    
    # 开始训练计时
    start_time = time.time()
    
    # 使用数据并行训练
    trained_model = train_parallel(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
        device_ids=device_ids
    )
    
    # 结束训练计时
    end_time = time.time()
    print(f"数据并行训练完成，耗时: {end_time - start_time:.2f}秒")
    
    # 验证模型
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        output = trained_model(test_input)
        print(f"测试输入输出示例: {output}")

# 示例2: 使用流水线并行训练模型
def example_pipeline_parallel():
    """
    流水线并行示例
    """
    print("\n===== 流水线并行示例 =====")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(num_samples=10000, input_dim=10, output_dim=5)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 创建较深的模型，适合流水线并行
    model = DeepModel(input_dim=10, hidden_dim=64, output_dim=5, num_layers=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 获取可用设备
    device_ids = list(range(min(4, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"使用设备: {device_ids}")
    
    # 流水线块大小
    chunk_size = 32
    
    # 开始训练计时
    start_time = time.time()
    
    # 使用流水线并行训练
    trained_model = train_pipeline(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
        device_ids=device_ids,
        chunk_size=chunk_size
    )
    
    # 结束训练计时
    end_time = time.time()
    print(f"流水线并行训练完成，耗时: {end_time - start_time:.2f}秒")
    
    # 验证模型
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        output = trained_model(test_input)
        print(f"测试输入输出示例: {output}")

# 示例3: 使用张量并行训练模型
def example_tensor_parallel():
    """
    张量并行示例
    """
    print("\n===== 张量并行示例 =====")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(num_samples=10000, input_dim=10, output_dim=5)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 创建模型、损失函数和优化器
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 获取可用设备
    device_ids = list(range(min(2, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"使用设备: {device_ids}")
    
    # 张量分割维度 (0=按行分割，1=按列分割)
    split_dim = 0
    
    # 开始训练计时
    start_time = time.time()
    
    # 使用张量并行训练
    trained_model = train_tensor_parallel(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
        device_ids=device_ids,
        split_dim=split_dim
    )
    
    # 结束训练计时
    end_time = time.time()
    print(f"张量并行训练完成，耗时: {end_time - start_time:.2f}秒")
    
    # 验证模型
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        output = trained_model(test_input)
        print(f"测试输入输出示例: {output}")

# 示例4: 比较三种并行策略的性能
def example_compare_parallel_strategies():
    """
    比较三种并行策略的性能
    """
    print("\n===== 比较三种并行策略性能 =====")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(num_samples=20000, input_dim=20, output_dim=10)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 获取可用设备
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device_ids = list(range(min(2, device_count)))
    print(f"可用设备数量: {device_count}")
    print(f"使用设备: {device_ids}")
    
    # 定义比较函数
    def benchmark_parallel_strategy(strategy_name, train_func, **kwargs):
        # 创建新模型和优化器
        model = SimpleModel(input_dim=20, hidden_dim=128, output_dim=10)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 开始计时
        start_time = time.time()
        
        # 训练模型
        train_func(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=2,
            **kwargs
        )
        
        # 结束计时
        end_time = time.time()
        
        print(f"{strategy_name} 训练耗时: {end_time - start_time:.2f}秒")
        
        return end_time - start_time
    
    # 运行基准测试
    dp_time = benchmark_parallel_strategy("数据并行", train_parallel, device_ids=device_ids)
    
    # 流水线并行需要较深的模型
    pp_model = DeepModel(input_dim=20, hidden_dim=128, output_dim=10, num_layers=8)
    pp_time = benchmark_parallel_strategy("流水线并行", train_pipeline, device_ids=device_ids, chunk_size=64)
    
    tp_time = benchmark_parallel_strategy("张量并行", train_tensor_parallel, device_ids=device_ids, split_dim=0)
    
    # 输出性能比较
    print("\n性能比较:")
    print(f"数据并行: {dp_time:.2f}秒")
    print(f"流水线并行: {pp_time:.2f}秒")
    print(f"张量并行: {tp_time:.2f}秒")
    
    # 找出最快的策略
    fastest = min(dp_time, pp_time, tp_time)
    if fastest == dp_time:
        print("数据并行性能最佳")
    elif fastest == pp_time:
        print("流水线并行性能最佳")
    else:
        print("张量并行性能最佳")

# 示例5: 使用ParallelLinear层进行张量并行
def example_parallel_linear():
    """
    使用ParallelLinear层进行张量并行的示例
    """
    print("\n===== ParallelLinear层示例 =====")
    
    # 获取可用设备
    device_ids = list(range(min(2, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"使用设备: {device_ids}")
    
    # 创建一个ParallelLinear层
    in_features = 100
    out_features = 50
    parallel_linear = ParallelLinear(in_features, out_features, device_ids, split_dim=0)
    
    # 创建输入张量
    input_tensor = torch.randn(32, in_features)
    
    # 前向传播
    output = parallel_linear(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证结果与标准线性层是否一致
    if len(device_ids) == 1:
        # 单设备情况下应与标准线性层结果相同
        standard_linear = nn.Linear(in_features, out_features)
        standard_linear.load_state_dict(parallel_linear.state_dict())
        
        with torch.no_grad():
            standard_output = standard_linear(input_tensor)
            diff = torch.mean(torch.abs(output - standard_output))
            print(f"与标准线性层的差异: {diff:.10f}")

# 示例6: 自定义并行策略组合
def example_custom_parallel_combination():
    """
    自定义组合不同并行策略的示例
    """
    print("\n===== 自定义并行策略组合示例 =====")
    
    # 此示例展示了如何根据实际需求组合不同的并行策略
    # 在真实场景中，您可能需要根据模型结构和硬件环境进行更复杂的设计
    
    # 获取可用设备
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"可用设备数量: {device_count}")
    
    if device_count >= 4:
        print("设备足够，可以尝试复杂的并行组合")
        print("例如: 数据并行 + 流水线并行 + 张量并行")
        print("- 将模型按层分割到不同设备组（流水线并行）")
        print("- 在每个设备组内使用数据并行处理不同批次")
        print("- 对大型张量使用张量并行进行协同计算")
    elif device_count >= 2:
        print("设备数量适中，建议尝试两种并行策略的组合")
        print("例如: 数据并行 + 张量并行")
        print("- 使用数据并行处理不同批次")
        print("- 对大型层使用张量并行加速计算")
    else:
        print("只有单个设备，无法进行真正的并行训练")
        print("建议: 可以使用此代码进行学习和测试，了解并行策略的工作原理")

# 运行所有示例
def run_all_examples():
    """
    运行所有示例
    """
    print("开始运行并行训练示例...")
    
    # 检查是否有GPU可用
    if torch.cuda.device_count() == 0:
        print("警告: 未检测到可用的GPU，所有并行策略将在CPU上模拟运行")
        print("注意: 在CPU上运行并行策略不会带来性能提升，反而可能更慢")
    
    # 运行各个示例
    example_data_parallel()
    example_pipeline_parallel()
    example_tensor_parallel()
    
    # 只有在有足够设备时才运行性能比较
    if torch.cuda.device_count() >= 2:
        example_compare_parallel_strategies()
    
    example_parallel_linear()
    example_custom_parallel_combination()
    
    print("\n所有示例运行完成！")

# 如果直接运行此文件，则执行所有示例
if __name__ == "__main__":
    run_all_examples()