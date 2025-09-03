from tensor import Tensor
import numpy as np

# 基本数学运算
def add(x, y):
    """张量加法"""
    return x + y

def sub(x, y):
    """张量减法"""
    return x - y

def mul(x, y):
    """张量乘法"""
    return x * y

def matmul(x, y):
    """矩阵乘法"""
    return x @ y

def neg(x):
    """取负运算"""
    return -x

def sum(x, axis=None, keepdims=False):
    """求和运算"""
    return x.sum(axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    """均值运算"""
    return x.mean(axis=axis, keepdims=keepdims)

# 激活函数
def relu(x):
    """ReLU激活函数"""
    result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def relu_backward(grad_output):
            grad = np.where(x.data > 0, 1, 0) * grad_output.data
            x.backward(Tensor(grad))
        
        result.grad_fn = type('ReLUBackward', (), {'backward': relu_backward})
    
    return result

def sigmoid(x):
    """Sigmoid激活函数"""
    sigmoid_val = 1. / (1. + np.exp(-x.data))
    result = Tensor(sigmoid_val, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def sigmoid_backward(grad_output):
            grad = sigmoid_val * (1 - sigmoid_val) * grad_output.data
            x.backward(Tensor(grad))
        
        result.grad_fn = type('SigmoidBackward', (), {'backward': sigmoid_backward})
    
    return result

def tanh(x):
    """Tanh激活函数"""
    tanh_val = np.tanh(x.data)
    result = Tensor(tanh_val, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def tanh_backward(grad_output):
            grad = (1 - tanh_val**2) * grad_output.data
            x.backward(Tensor(grad))
        
        result.grad_fn = type('TanhBackward', (), {'backward': tanh_backward})
    
    return result

def softmax(x, axis=-1):
    """Softmax函数"""
    # 防止数值不稳定，减去最大值
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_val = exp_x / sum_exp
    
    result = Tensor(softmax_val, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def softmax_backward(grad_output):
            # softmax的梯度：softmax * (grad - sum(softmax * grad, axis=axis, keepdims=True))
            s = softmax_val
            grad = s * (grad_output.data - np.sum(s * grad_output.data, axis=axis, keepdims=True))
            x.backward(Tensor(grad))
        
        result.grad_fn = type('SoftmaxBackward', (), {'backward': softmax_backward})
    
    return result

# 损失函数
def mse_loss(input, target):
    """均方误差损失"""
    if isinstance(target, Tensor):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    error = input.data - target_data
    loss = np.mean(error ** 2)
    
    result = Tensor(loss, requires_grad=input.requires_grad)
    
    if input.requires_grad:
        def mse_backward(grad_output):
            grad = (2 * error / error.size) * grad_output.data
            input.backward(Tensor(grad))
        
        result.grad_fn = type('MSELossBackward', (), {'backward': mse_backward})
    
    return result

def cross_entropy_loss(input, target):
    """交叉熵损失"""
    if isinstance(target, Tensor):
        target_data = target.data
    else:
        target_data = np.array(target)
    
    # 防止数值不稳定
    x_max = np.max(input.data, axis=-1, keepdims=True)
    exp_x = np.exp(input.data - x_max)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    log_softmax = (input.data - x_max) - np.log(sum_exp)
    
    if target_data.ndim == 1:
        batch_size = input.data.shape[0]
        loss = -np.sum(log_softmax[np.arange(batch_size), target_data]) / batch_size
    else:
        loss = -np.mean(np.sum(target_data * log_softmax, axis=-1))
    
    result = Tensor(loss, requires_grad=input.requires_grad)
    
    if input.requires_grad:
        def cross_entropy_backward(grad_output):
            softmax = exp_x / sum_exp
            
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

# 张量操作
def reshape(x, shape):
    """重塑张量形状"""
    result = Tensor(x.data.reshape(shape), requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def reshape_backward(grad_output):
            grad = grad_output.data.reshape(x.shape)
            x.backward(Tensor(grad))
        
        result.grad_fn = type('ReshapeBackward', (), {'backward': reshape_backward})
    
    return result

def transpose(x, dim0, dim1):
    """交换张量的两个维度"""
    result = Tensor(np.transpose(x.data, (dim0, dim1) if dim0 == 0 and dim1 == 1 else 
                                tuple(i for i in range(x.ndim) if i != dim0 and i != dim1) + (dim0, dim1)), 
                   requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def transpose_backward(grad_output):
            # 再次交换维度以恢复原始形状
            grad = np.transpose(grad_output.data, (dim0, dim1) if dim0 == 0 and dim1 == 1 else 
                               tuple(i for i in range(grad_output.ndim) if i != dim0 and i != dim1) + (dim1, dim0))
            x.backward(Tensor(grad))
        
        result.grad_fn = type('TransposeBackward', (), {'backward': transpose_backward})
    
    return result

def expand_dims(x, axis):
    """在指定位置添加维度"""
    result = Tensor(np.expand_dims(x.data, axis), requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def expand_dims_backward(grad_output):
            # 删除添加的维度
            grad = np.squeeze(grad_output.data, axis=axis)
            x.backward(Tensor(grad))
        
        result.grad_fn = type('ExpandDimsBackward', (), {'backward': expand_dims_backward})
    
    return result

def squeeze(x, axis=None):
    """删除大小为1的维度"""
    result = Tensor(np.squeeze(x.data, axis=axis), requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def squeeze_backward(grad_output):
            # 恢复被删除的维度
            if axis is None:
                # 找出原始数据中所有大小为1的维度
                orig_shape = list(x.shape)
                grad_shape = list(grad_output.shape)
                new_shape = []
                g_idx = 0
                for s in orig_shape:
                    if s == 1:
                        new_shape.append(1)
                    else:
                        new_shape.append(grad_shape[g_idx])
                        g_idx += 1
                grad = grad_output.data.reshape(new_shape)
            else:
                if isinstance(axis, int):
                    axis = (axis,)
                # 在指定位置插入大小为1的维度
                grad = np.expand_dims(grad_output.data, axis=axis)
            x.backward(Tensor(grad))
        
        result.grad_fn = type('SqueezeBackward', (), {'backward': squeeze_backward})
    
    return result

# 快捷导入
def F():
    """返回函数式API对象，方便链式调用"""
    class FunctionalAPI:
        def __init__(self):
            self.add = add
            self.sub = sub
            self.mul = mul
            self.matmul = matmul
            self.neg = neg
            self.sum = sum
            self.mean = mean
            self.relu = relu
            self.sigmoid = sigmoid
            self.tanh = tanh
            self.softmax = softmax
            self.mse_loss = mse_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reshape = reshape
            self.transpose = transpose
            self.expand_dims = expand_dims
            self.squeeze = squeeze
    
    return FunctionalAPI()