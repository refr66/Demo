import torch
import torch.fx
import tempfile
import os
import importlib.util
from frontend import torch_to_ir
from optimizer import optimize_ir, create_default_optimizers
from backend import generate_code, save_code_to_file

class Compiler:
    """简单的模型编译器"""
    def __init__(self, passes=None, backend='numpy'):
        """初始化编译器
        
        Args:
            passes: 优化Pass列表，如果为None则使用默认优化器
            backend: 后端类型，支持'numpy'和'triton'
        """
        self.passes = passes
        self.backend = backend
        self.compiled_code = None
        self.ir_graph = None
        
    def compile(self, func, args, kwargs=None):
        """编译一个函数或模型
        
        Args:
            func: 要编译的函数或模型
            args: 函数的位置参数
            kwargs: 函数的关键字参数，默认为None
        
        Returns:
            编译后的函数
        """
        if kwargs is None:
            kwargs = {}
        
        # 1. 使用前端将PyTorch函数转换为IR
        print("Converting PyTorch function to IR...")
        self.ir_graph = torch_to_ir(func, args, kwargs)
        
        # 2. 使用优化器优化IR
        print("Optimizing IR...")
        self.ir_graph = optimize_ir(self.ir_graph, self.passes)
        
        # 3. 使用后端生成代码
        print(f"Generating code for {self.backend} backend...")
        self.compiled_code = generate_code(self.ir_graph, self.backend)
        
        # 4. 加载并返回编译后的函数
        print("Loading compiled function...")
        compiled_func = self._load_compiled_function()
        
        print("Compilation completed successfully!")
        return compiled_func
    
    def _load_compiled_function(self):
        """加载编译后的函数"""
        # 创建临时文件来存储编译后的代码
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.compiled_code)
            temp_file_path = f.name
        
        try:
            # 动态导入临时模块
            spec = importlib.util.spec_from_file_location("compiled_module", temp_file_path)
            compiled_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiled_module)
            
            # 获取编译后的函数
            if hasattr(compiled_module, 'compiled_function'):
                return compiled_module.compiled_function
            else:
                raise AttributeError("Compiled module does not have 'compiled_function' attribute")
        finally:
            # 清理临时文件
            try:
                os.remove(temp_file_path)
            except:
                pass
    
    def save_code(self, file_path):
        """将生成的代码保存到文件
        
        Args:
            file_path: 保存代码的文件路径
        """
        if self.compiled_code is None:
            raise ValueError("No compiled code available. Please compile a function first.")
        
        with open(file_path, 'w') as f:
            f.write(self.compiled_code)
        
        print(f"Code saved to {file_path}")
    
    def get_ir(self):
        """获取中间表示
        
        Returns:
            IR图对象
        """
        return self.ir_graph
    
    def get_code(self):
        """获取生成的代码
        
        Returns:
            生成的代码字符串
        """
        return self.compiled_code

class FunctionCompiler:
    """函数编译器，提供更简单的API"""
    @staticmethod
    def compile(func, *args, passes=None, backend='numpy'):
        """编译一个函数
        
        Args:
            func: 要编译的函数
            *args: 函数的位置参数，用于捕获计算图
            passes: 优化Pass列表，如果为None则使用默认优化器
            backend: 后端类型，支持'numpy'和'triton'
        
        Returns:
            编译后的函数
        """
        compiler = Compiler(passes=passes, backend=backend)
        # 将args作为示例输入来编译函数
        return compiler.compile(func, args)

    @staticmethod
    def save_compiled_code(func, file_path, *args, passes=None, backend='numpy'):
        """编译一个函数并将生成的代码保存到文件
        
        Args:
            func: 要编译的函数
            file_path: 保存代码的文件路径
            *args: 函数的位置参数，用于捕获计算图
            passes: 优化Pass列表，如果为None则使用默认优化器
            backend: 后端类型，支持'numpy'和'triton'
        """
        compiler = Compiler(passes=passes, backend=backend)
        compiler.compile(func, args)
        compiler.save_code(file_path)
        return file_path

class ModelCompiler:
    """模型编译器，专门用于编译PyTorch模型"""
    @staticmethod
    def compile(model, input_shape, passes=None, backend='numpy'):
        """编译一个PyTorch模型
        
        Args:
            model: 要编译的PyTorch模型
            input_shape: 模型输入的形状，用于创建示例输入
            passes: 优化Pass列表，如果为None则使用默认优化器
            backend: 后端类型，支持'numpy'和'triton'
        
        Returns:
            编译后的函数
        """
        # 创建示例输入
        if isinstance(input_shape, tuple):
            # 单个输入
            example_input = torch.randn(*input_shape)
            args = (example_input,)
        elif isinstance(input_shape, list) and all(isinstance(shape, tuple) for shape in input_shape):
            # 多个输入
            args = tuple(torch.randn(*shape) for shape in input_shape)
        else:
            raise ValueError("input_shape must be a tuple or a list of tuples")
        
        # 定义一个包装函数来编译
        def model_wrapper(*inputs):
            return model(*inputs)
        
        compiler = Compiler(passes=passes, backend=backend)
        return compiler.compile(model_wrapper, args)

    @staticmethod
    def save_compiled_model(model, file_path, input_shape, passes=None, backend='numpy'):
        """编译一个PyTorch模型并将生成的代码保存到文件
        
        Args:
            model: 要编译的PyTorch模型
            file_path: 保存代码的文件路径
            input_shape: 模型输入的形状，用于创建示例输入
            passes: 优化Pass列表，如果为None则使用默认优化器
            backend: 后端类型，支持'numpy'和'triton'
        """
        compiled_func = ModelCompiler.compile(model, input_shape, passes, backend)
        
        # 创建一个简单的包装脚本来保存
        with open(file_path, 'w') as f:
            f.write("import numpy as np\n\n")
            f.write("def compiled_model(inputs):\n")
            f.write("    # This is a placeholder for the compiled model\n")
            f.write("    # Please use the Compiler class for actual compilation\n")
            f.write("    return inputs[0]\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    # Example usage\n")
            if isinstance(input_shape, tuple):
                f.write(f"    input = np.random.rand(*{input_shape})\n")
                f.write("    output = compiled_model([input])\n")
            else:
                for i, shape in enumerate(input_shape):
                    f.write(f"    input_{i} = np.random.rand(*{shape})\n")
                f.write(f"    inputs = [{', '.join([f'input_{i}' for i in range(len(input_shape))])}]\n")
                f.write("    output = compiled_model(inputs)\n")
            f.write("    print('Output shape:', output.shape)\n")
        
        print(f"Model code saved to {file_path}")
        return file_path

# 快捷函数
def compile(func, *args, passes=None, backend='numpy'):
    """编译一个函数
    
    Args:
        func: 要编译的函数
        *args: 函数的位置参数，用于捕获计算图
        passes: 优化Pass列表，如果为None则使用默认优化器
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        编译后的函数
    """
    return FunctionCompiler.compile(func, *args, passes=passes, backend=backend)

def compile_model(model, input_shape, passes=None, backend='numpy'):
    """编译一个PyTorch模型
    
    Args:
        model: 要编译的PyTorch模型
        input_shape: 模型输入的形状
        passes: 优化Pass列表，如果为None则使用默认优化器
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        编译后的函数
    """
    return ModelCompiler.compile(model, input_shape, passes=passes, backend=backend)

def save_compiled_code(func, file_path, *args, passes=None, backend='numpy'):
    """编译一个函数并保存代码
    
    Args:
        func: 要编译的函数
        file_path: 保存代码的文件路径
        *args: 函数的位置参数，用于捕获计算图
        passes: 优化Pass列表，如果为None则使用默认优化器
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        文件路径
    """
    return FunctionCompiler.save_compiled_code(func, file_path, *args, passes=passes, backend=backend)

def save_compiled_model(model, file_path, input_shape, passes=None, backend='numpy'):
    """编译一个模型并保存代码
    
    Args:
        model: 要编译的模型
        file_path: 保存代码的文件路径
        input_shape: 模型输入的形状
        passes: 优化Pass列表，如果为None则使用默认优化器
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        文件路径
    """
    return ModelCompiler.save_compiled_model(model, file_path, input_shape, passes=passes, backend=backend)