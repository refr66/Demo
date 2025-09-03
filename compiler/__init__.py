"""简单的模型编译器：将PyTorch计算图转换为优化的Python/NumPy或Triton代码"""

# 版本信息
__version__ = "0.1.0"

# 导出核心功能
from .compiler import (
    Compiler,
    FunctionCompiler,
    ModelCompiler,
    compile,
    compile_model,
    save_compiled_code,
    save_compiled_model
)

from .frontend import (
    torch_to_ir,
    PyTorchFrontend
)

from .optimizer import (
    GraphOptimizer,
    PassManager,
    create_default_optimizers,
    optimize_ir
)

from .backend import (
    CodeGenerator,
    PythonNumpyBackend,
    TritonBackend,
    generate_code,
    save_code_to_file
)

from .ir import (
    IRNode,
    TensorNode,
    ParameterNode,
    AddNode,
    MatmulNode,
    ReLUNode,
    SigmoidNode,
    TanhNode,
    FusedAddReluNode,
    FusedMatmulAddNode,
    IRGraph
)

# 快捷访问函数
def get_compiler(backend='numpy'):
    """获取一个编译器实例
    
    Args:
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        Compiler实例
    """
    return Compiler(backend=backend)

def get_frontend():
    """获取一个前端实例
    
    Returns:
        PyTorchFrontend实例
    """
    return PyTorchFrontend()

def get_optimizer():
    """获取一个默认的优化器Pass管理器
    
    Returns:
        PassManager实例
    """
    return create_default_optimizers()

def get_backend(ir_graph, backend='numpy'):
    """获取一个后端代码生成器实例
    
    Args:
        ir_graph: IR图对象
        backend: 后端类型，支持'numpy'和'triton'
    
    Returns:
        CodeGenerator实例
    """
    if backend.lower() == 'triton':
        return TritonBackend(ir_graph)
    else:
        return PythonNumpyBackend(ir_graph)

# 包描述
__doc__ = """
简单模型编译器
==============

这是一个简单的模型编译器，能够将PyTorch计算图转换为优化的Python/NumPy或Triton代码。

主要特性：
--------
- 前端：使用PyTorch的FX模块捕获计算图并转换为自定义IR
- 优化：支持多种图优化pass，如算子融合、死代码消除、常量折叠等
- 后端：支持生成Python/NumPy代码和Triton代码

使用方法：
--------

1. 编译简单函数：
```python
import torch
from compiler import compile

def simple_function(x, w, b):
    return torch.relu(torch.matmul(x, w) + b)

# 创建示例输入
x = torch.randn(10, 20)
w = torch.randn(20, 5)
b = torch.randn(5)

# 编译函数
compiled_func = compile(simple_function, x, w, b)

# 运行编译后的函数
import numpy as np
numpy_inputs = [x.numpy(), w.numpy(), b.numpy()]
output = compiled_func(numpy_inputs)
```

2. 编译PyTorch模型：
```python
import torch
from compiler import compile_model

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
```

3. 保存编译后的代码：
```python
from compiler import save_compiled_code

save_compiled_code(simple_function, "compiled_function.py", x, w, b)
```

4. 使用不同的后端：
```python
# 使用NumPy后端编译
compiled_numpy = compile(simple_function, x, w, b, backend='numpy')

# 使用Triton后端编译
compiled_triton = compile(simple_function, x, w, b, backend='triton')
```

5. 使用自定义优化器：
```python
from compiler import create_default_optimizers

# 创建默认优化器
optimizers = create_default_optimizers()

# 编译函数时使用自定义优化器
compiled_func = compile(simple_function, x, w, b, passes=optimizers)
"""

# 导出所有公共API
__all__ = [
    # 编译器类
    'Compiler',
    'FunctionCompiler',
    'ModelCompiler',
    
    # 编译函数
    'compile',
    'compile_model',
    'save_compiled_code',
    'save_compiled_model',
    
    # 前端相关
    'torch_to_ir',
    'PyTorchFrontend',
    
    # 优化器相关
    'GraphOptimizer',
    'PassManager',
    'create_default_optimizers',
    'optimize_ir',
    
    # 后端相关
    'CodeGenerator',
    'PythonNumpyBackend',
    'TritonBackend',
    'generate_code',
    'save_code_to_file',
    
    # IR相关
    'IRNode',
    'TensorNode',
    'ParameterNode',
    'AddNode',
    'MatmulNode',
    'ReLUNode',
    'SigmoidNode',
    'TanhNode',
    'FusedAddReluNode',
    'FusedMatmulAddNode',
    'IRGraph',
    
    # 快捷访问函数
    'get_compiler',
    'get_frontend',
    'get_optimizer',
    'get_backend',
    
    # 版本信息
    '__version__',
]