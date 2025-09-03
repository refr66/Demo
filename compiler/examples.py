import torch
import numpy as np
import time
from compiler import compile, compile_model, save_compiled_code
from optimizer import create_default_optimizers

# 示例1：简单的函数编译
def example_simple_function():
    """示例：编译一个简单的函数"""
    print("=== 示例1：简单的函数编译 ===")
    
    # 定义一个简单的函数
    def simple_function(x, w, b):
        return torch.relu(torch.matmul(x, w) + b)
    
    # 创建示例输入
    x = torch.randn(10, 20)
    w = torch.randn(20, 5)
    b = torch.randn(5)
    
    # 编译函数
    compiled_func = compile(simple_function, x, w, b)
    
    # 运行原始函数和编译后的函数
    start_time = time.time()
    original_output = simple_function(x, w, b)
    original_time = time.time() - start_time
    
    # 将PyTorch张量转换为NumPy数组作为输入
    numpy_inputs = [x.numpy(), w.numpy(), b.numpy()]
    
    start_time = time.time()
    compiled_output = compiled_func(numpy_inputs)
    compiled_time = time.time() - start_time
    
    # 检查结果是否匹配
    np.testing.assert_allclose(original_output.detach().numpy(), compiled_output, rtol=1e-6)
    
    print(f"原始函数执行时间: {original_time:.6f}秒")
    print(f"编译后函数执行时间: {compiled_time:.6f}秒")
    print(f"加速比: {original_time/compiled_time:.2f}x")
    print("结果验证通过：编译后的函数输出与原始函数一致")
    print()

# 示例2：使用不同的后端
def example_different_backends():
    """示例：使用不同的后端编译函数"""
    print("=== 示例2：使用不同的后端编译函数 ===")
    
    # 定义一个简单的函数
    def simple_function(x):
        return torch.relu(x)
    
    # 创建示例输入
    x = torch.randn(1000, 1000)
    
    try:
        # 使用NumPy后端编译
        print("使用NumPy后端编译...")
        compiled_numpy = compile(simple_function, x, backend='numpy')
        
        # 运行编译后的函数
        numpy_input = [x.numpy()]
        numpy_output = compiled_numpy(numpy_input)
        print("NumPy后端编译成功！")
        
        # 尝试使用Triton后端编译（如果可用）
        print("尝试使用Triton后端编译...")
        compiled_triton = compile(simple_function, x, backend='triton')
        
        # 运行编译后的函数
        triton_input = [x.numpy().astype(np.float32)]  # Triton通常使用float32
        triton_output = compiled_triton(triton_input)
        print("Triton后端编译成功！")
        
        # 检查结果是否匹配
        np.testing.assert_allclose(numpy_output, triton_output, rtol=1e-6)
        print("结果验证通过：不同后端的输出一致")
    except ImportError as e:
        print(f"Triton不可用: {e}")
    except Exception as e:
        print(f"编译过程中出错: {e}")
    
    print()

# 示例3：编译一个简单的PyTorch模型
def example_compile_model():
    """示例：编译一个简单的PyTorch模型"""
    print("=== 示例3：编译一个简单的PyTorch模型 ===")
    
    # 定义一个简单的模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型实例
    model = SimpleModel()
    
    # 编译模型
    input_shape = (32, 10)  # 批次大小为32，输入特征为10
    compiled_model = compile_model(model, input_shape)
    
    # 创建示例输入
    torch_input = torch.randn(*input_shape)
    numpy_input = [torch_input.numpy()]
    
    # 运行原始模型和编译后的模型
    with torch.no_grad():
        start_time = time.time()
        original_output = model(torch_input)
        original_time = time.time() - start_time
    
    start_time = time.time()
    compiled_output = compiled_model(numpy_input)
    compiled_time = time.time() - start_time
    
    # 检查结果是否匹配
    np.testing.assert_allclose(original_output.numpy(), compiled_output, rtol=1e-6)
    
    print(f"原始模型执行时间: {original_time:.6f}秒")
    print(f"编译后模型执行时间: {compiled_time:.6f}秒")
    print(f"加速比: {original_time/compiled_time:.2f}x")
    print("结果验证通过：编译后的模型输出与原始模型一致")
    print()

# 示例4：保存编译后的代码
def example_save_compiled_code():
    """示例：保存编译后的代码"""
    print("=== 示例4：保存编译后的代码 ===")
    
    # 定义一个简单的函数
    def simple_function(x, w):
        return torch.matmul(x, w)
    
    # 创建示例输入
    x = torch.randn(10, 20)
    w = torch.randn(20, 5)
    
    # 保存编译后的代码到文件
    file_path = "compiled_function.py"
    save_compiled_code(simple_function, file_path, x, w)
    
    print(f"编译后的代码已保存到: {file_path}")
    
    # 显示文件的前几行
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[:10]
            print("\n编译后的代码预览:")
            for line in lines:
                print(line.strip())
            if len(lines) >= 10:
                print("...")
    except Exception as e:
        print(f"无法读取文件: {e}")
    
    print()

# 示例5：使用自定义优化器
def example_custom_optimizers():
    """示例：使用自定义优化器"""
    print("=== 示例5：使用自定义优化器 ===")
    
    # 定义一个简单的函数
    def function_with_fusion_opportunities(x, w, b):
        # 这个函数有两个可以融合的操作：matmul+add和add+relu
        y = torch.matmul(x, w) + b  # 可以融合为matmul_add
        z = y + torch.relu(y)  # 可以融合为add_relu
        return z
    
    # 创建示例输入
    x = torch.randn(10, 20)
    w = torch.randn(20, 5)
    b = torch.randn(5)
    
    # 创建默认优化器
    optimizers = create_default_optimizers()
    
    # 编译函数时使用自定义优化器
    compiled_func = compile(function_with_fusion_opportunities, x, w, b, passes=optimizers)
    
    # 运行编译后的函数
    numpy_inputs = [x.numpy(), w.numpy(), b.numpy()]
    compiled_output = compiled_func(numpy_inputs)
    
    # 运行原始函数
    original_output = function_with_fusion_opportunities(x, w, b)
    
    # 检查结果是否匹配
    np.testing.assert_allclose(original_output.detach().numpy(), compiled_output, rtol=1e-6)
    
    print("结果验证通过：使用自定义优化器编译后的函数输出与原始函数一致")
    print()

# 示例6：性能比较 - 不同输入大小
def example_performance_comparison():
    """示例：不同输入大小的性能比较"""
    print("=== 示例6：不同输入大小的性能比较 ===")
    
    # 定义一个简单的函数
    def performance_function(x, w, b):
        return torch.relu(torch.matmul(x, w) + b)
    
    # 不同的输入大小
    sizes = [(100, 100, 100), (1000, 1000, 1000), (2000, 2000, 2000)]
    
    print(f"{'输入大小':<20}{'原始函数(秒)':<15}{'编译后(秒)':<15}{'加速比':<10}")
    print("=" * 60)
    
    for size in sizes:
        # 创建示例输入
        x = torch.randn(size[0], size[1])
        w = torch.randn(size[1], size[2])
        b = torch.randn(size[2])
        
        # 编译函数
        compiled_func = compile(performance_function, x, w, b)
        
        # 运行原始函数
        start_time = time.time()
        performance_function(x, w, b)
        original_time = time.time() - start_time
        
        # 运行编译后的函数
        numpy_inputs = [x.numpy(), w.numpy(), b.numpy()]
        start_time = time.time()
        compiled_func(numpy_inputs)
        compiled_time = time.time() - start_time
        
        # 计算加速比
        speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
        
        print(f"{size:<20}{original_time:.6f}      {compiled_time:.6f}      {speedup:.2f}x")
    
    print()

# 运行所有示例
def run_all_examples():
    """运行所有示例"""
    print("==================== 简单模型编译器示例 ====================")
    print()
    
    example_simple_function()
    example_different_backends()
    example_compile_model()
    example_save_compiled_code()
    example_custom_optimizers()
    example_performance_comparison()
    
    print("==================== 所有示例运行完成 ====================")

# 如果直接运行此文件，则运行所有示例
if __name__ == '__main__':
    run_all_examples()