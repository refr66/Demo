from tensor import Tensor, zeros, ones, randn
import numpy as np

class Module:
    """所有神经网络模块的基类"""
    def __init__(self):
        self.training = True
        self.parameters = []
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        for param in self.parameters:
            if isinstance(param, Module):
                param.train()
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
        for param in self.parameters:
            if isinstance(param, Module):
                param.eval()
    
    def forward(self, x):
        """前向传播，需要被子类重写"""
        raise NotImplementedError
    
    def __call__(self, x):
        """调用模块进行前向传播"""
        return self.forward(x)
    
    def add_parameter(self, param):
        """添加参数到模块"""
        self.parameters.append(param)

class Linear(Module):
    """线性变换层 (全连接层)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # 使用Xavier初始化权重
        stdv = 1. / np.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)), requires_grad=True)
        
        self.bias = None
        if bias:
            self.bias = Tensor(np.random.uniform(-stdv, stdv, (out_features,)), requires_grad=True)
        
        self.add_parameter(self.weight)
        if self.bias is not None:
            self.add_parameter(self.bias)
    
    def forward(self, x):
        """前向传播：y = x @ weight^T + bias"""
        output = x @ self.weight.data.T
        if self.bias is not None:
            output = output + self.bias.data
        return Tensor(output, requires_grad=x.requires_grad)

class ReLU(Module):
    """ReLU激活函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """前向传播：ReLU(x) = max(0, x)"""
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def relu_backward(grad_output):
                # ReLU的导数：x>0时为1，否则为0
                grad = np.where(x.data > 0, 1, 0) * grad_output.data
                x.backward(Tensor(grad))
            
            result.grad_fn = type('ReLUBackward', (), {'backward': relu_backward})
        
        return result

class Sigmoid(Module):
    """Sigmoid激活函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """前向传播：sigmoid(x) = 1 / (1 + exp(-x))"""
        sigmoid_val = 1. / (1. + np.exp(-x.data))
        result = Tensor(sigmoid_val, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def sigmoid_backward(grad_output):
                # Sigmoid的导数：sigmoid(x) * (1 - sigmoid(x))
                grad = sigmoid_val * (1 - sigmoid_val) * grad_output.data
                x.backward(Tensor(grad))
            
            result.grad_fn = type('SigmoidBackward', (), {'backward': sigmoid_backward})
        
        return result

class Tanh(Module):
    """Tanh激活函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """前向传播：tanh(x)"""
        tanh_val = np.tanh(x.data)
        result = Tensor(tanh_val, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def tanh_backward(grad_output):
                # Tanh的导数：1 - tanh(x)^2
                grad = (1 - tanh_val**2) * grad_output.data
                x.backward(Tensor(grad))
            
            result.grad_fn = type('TanhBackward', (), {'backward': tanh_backward})
        
        return result

class Sequential(Module):
    """按顺序组合多个模块"""
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        for module in modules:
            self.add_parameter(module)
    
    def forward(self, x):
        """前向传播：依次通过每个模块"""
        for module in self.modules:
            x = module(x)
        return x

class MSELoss(Module):
    """均方误差损失函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        """前向传播：计算均方误差"""
        if isinstance(target, Tensor):
            target_data = target.data
        else:
            target_data = np.array(target)
        
        # 计算误差平方
        error = input.data - target_data
        squared_error = error ** 2
        
        # 计算均值
        loss = np.mean(squared_error)
        result = Tensor(loss, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def mse_backward(grad_output):
                # MSE的导数：2*(input - target)/N
                grad = (2 * error / error.size) * grad_output.data
                input.backward(Tensor(grad))
            
            result.grad_fn = type('MSELossBackward', (), {'backward': mse_backward})
        
        return result

class CrossEntropyLoss(Module):
    """交叉熵损失函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        """前向传播：计算交叉熵损失"""
        if isinstance(target, Tensor):
            target_data = target.data
        else:
            target_data = np.array(target)
        
        # 防止数值不稳定，减去最大值
        input_max = np.max(input.data, axis=-1, keepdims=True)
        exp_input = np.exp(input.data - input_max)
        sum_exp = np.sum(exp_input, axis=-1, keepdims=True)
        log_softmax = (input.data - input_max) - np.log(sum_exp)
        
        # 如果目标是类别索引
        if target_data.ndim == 1:
            batch_size = input.data.shape[0]
            loss = -np.sum(log_softmax[np.arange(batch_size), target_data]) / batch_size
        # 如果目标是one-hot编码
        else:
            loss = -np.mean(np.sum(target_data * log_softmax, axis=-1))
        
        result = Tensor(loss, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def cross_entropy_backward(grad_output):
                # 交叉熵的导数：softmax(input) - target
                softmax = exp_input / sum_exp
                
                if target_data.ndim == 1:
                    batch_size = input.data.shape[0]
                    grad = softmax.copy()
                    grad[np.arange(batch_size), target_data] -= 1
                    grad /= batch_size
                else:
                    grad = softmax - target_data
                    grad /= input.data.shape[0]
                
                input.backward(Tensor(grad * grad_output.data))
            
            result.grad_fn = type('CrossEntropyLossBackward', (), {'backward': cross_entropy_backward})
        
        return result

class Dropout(Module):
    """Dropout层"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # dropout概率
        self.mask = None
    
    def forward(self, x):
        """前向传播：应用dropout"""
        if self.training and self.p > 0:
            # 创建掩码，保留概率为1-p
            self.mask = np.random.rand(*x.data.shape) > self.p
            # 缩放保留的元素以保持期望不变
            scaled_x = x.data * self.mask / (1 - self.p)
            result = Tensor(scaled_x, requires_grad=x.requires_grad)
        else:
            result = Tensor(x.data.copy(), requires_grad=x.requires_grad)
        
        if x.requires_grad and self.training and self.p > 0:
            def dropout_backward(grad_output):
                # 反向传播时应用相同的掩码
                grad = grad_output.data * self.mask / (1 - self.p)
                x.backward(Tensor(grad))
            
            result.grad_fn = type('DropoutBackward', (), {'backward': dropout_backward})
        
        return result

# 快捷访问
def linear(in_features, out_features, bias=True):
    return Linear(in_features, out_features, bias)

def relu():
    return ReLU()

def sigmoid():
    return Sigmoid()

def tanh():
    return Tanh()

def sequential(*modules):
    return Sequential(*modules)

def mse_loss():
    return MSELoss()

def cross_entropy_loss():
    return CrossEntropyLoss()

def dropout(p=0.5):
    return Dropout(p)