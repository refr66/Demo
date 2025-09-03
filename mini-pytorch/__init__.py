# Mini-PyTorch包初始化文件

# 导出核心类和函数
from .tensor import Tensor, tensor, zeros, ones, randn
from .nn import Module, Linear, ReLU, Sigmoid, Tanh, Sequential, MSELoss, CrossEntropyLoss, Dropout
from .optim import Optimizer, SGD, Adam
from . import functional as F


# 版本信息
__version__ = "0.1.0"

# 导出快捷访问
def linear(*args, **kwargs):
    return Linear(*args, **kwargs)

def relu():
    return ReLU()

def sigmoid():
    return Sigmoid()

def tanh():
    return Tanh()

def sequential(*args):
    return Sequential(*args)

def mse_loss():
    return MSELoss()

def cross_entropy_loss():
    return CrossEntropyLoss()

def dropout(*args, **kwargs):
    return Dropout(*args, **kwargs)

def sgd(*args, **kwargs):
    return SGD(*args, **kwargs)

def adam(*args, **kwargs):
    return Adam(*args, **kwargs)

# 包描述
__all__ = [
    'Tensor', 'tensor', 'zeros', 'ones', 'randn',
    'Module', 'Linear', 'ReLU', 'Sigmoid', 'Tanh', 'Sequential', 'MSELoss', 'CrossEntropyLoss', 'Dropout',
    'Optimizer', 'SGD', 'Adam',
    'F',
    'linear', 'relu', 'sigmoid', 'tanh', 'sequential', 'mse_loss', 'cross_entropy_loss', 'dropout',
    'sgd', 'adam',
    '__version__'
]