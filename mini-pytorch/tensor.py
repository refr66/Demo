import numpy as np

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype) if dtype else data
        elif isinstance(data, list):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.array([data], dtype=dtype)
        
        self.requires_grad = requires_grad
        self.grad = None  # 梯度
        self.grad_fn = None  # 梯度函数
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def backward(self, grad_output=None):
        """反向传播计算梯度"""
        if not self.requires_grad:
            return
        
        if grad_output is None:
            # 对于标量输出，默认梯度为1
            if self.data.size == 1:
                grad_output = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError("grad_output must be specified for non-scalar Tensors")
        
        # 存储梯度
        if self.grad is None:
            self.grad = grad_output.data.copy()
        else:
            self.grad += grad_output.data
        
        # 调用梯度函数继续反向传播
        if self.grad_fn is not None:
            self.grad_fn.backward(grad_output)
    
    def zero_grad(self):
        """清空梯度"""
        if self.grad is not None:
            self.grad.fill(0)
    
    def numpy(self):
        """转换为numpy数组"""
        return self.data.copy()
    
    def __add__(self, other):
        """加法运算"""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data + other_data, requires_grad=self.requires_grad)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            # 创建加法的梯度函数
            def add_backward(grad_output):
                if self.requires_grad:
                    self.backward(grad_output)
                if isinstance(other, Tensor) and other.requires_grad:
                    # 对于广播操作，需要对梯度进行求和
                    if grad_output.shape != other.shape:
                        grad = np.sum(grad_output.data, axis=tuple(range(len(grad_output.shape) - len(other.shape))), keepdims=True)
                        other.backward(Tensor(grad))
                    else:
                        other.backward(grad_output)
            
            result.grad_fn = type('AddBackward', (), {'backward': add_backward})
        
        return result
    
    def __sub__(self, other):
        """减法运算"""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data - other_data, requires_grad=self.requires_grad)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            def sub_backward(grad_output):
                if self.requires_grad:
                    self.backward(grad_output)
                if isinstance(other, Tensor) and other.requires_grad:
                    if grad_output.shape != other.shape:
                        grad = np.sum(-grad_output.data, axis=tuple(range(len(grad_output.shape) - len(other.shape))), keepdims=True)
                        other.backward(Tensor(grad))
                    else:
                        other.backward(Tensor(-grad_output.data))
            
            result.grad_fn = type('SubBackward', (), {'backward': sub_backward})
        
        return result
    
    def __mul__(self, other):
        """乘法运算"""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data * other_data, requires_grad=self.requires_grad)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            def mul_backward(grad_output):
                if self.requires_grad:
                    grad = grad_output.data * other_data
                    if grad.shape != self.shape:
                        grad = np.sum(grad, axis=tuple(range(len(grad.shape) - len(self.shape))), keepdims=True)
                    self.backward(Tensor(grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    grad = grad_output.data * self.data
                    if grad.shape != other.shape:
                        grad = np.sum(grad, axis=tuple(range(len(grad.shape) - len(other.shape))), keepdims=True)
                    other.backward(Tensor(grad))
            
            result.grad_fn = type('MulBackward', (), {'backward': mul_backward})
        
        return result
    
    def __matmul__(self, other):
        """矩阵乘法"""
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data @ other_data, requires_grad=self.requires_grad)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            def matmul_backward(grad_output):
                if self.requires_grad:
                    # dL/dA = grad_output @ B^T
                    grad = grad_output.data @ other_data.T
                    self.backward(Tensor(grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    # dL/dB = A^T @ grad_output
                    grad = self.data.T @ grad_output.data
                    other.backward(Tensor(grad))
            
            result.grad_fn = type('MatmulBackward', (), {'backward': matmul_backward})
        
        return result
    
    def __neg__(self):
        """取负运算"""
        result = Tensor(-self.data, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def neg_backward(grad_output):
                self.backward(Tensor(-grad_output.data))
            
            result.grad_fn = type('NegBackward', (), {'backward': neg_backward})
        
        return result
    
    def sum(self, axis=None, keepdims=False):
        """求和运算"""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def sum_backward(grad_output):
                # 广播梯度到原始形状
                if axis is not None:
                    if not isinstance(axis, tuple):
                        axis = (axis,)
                    expand_shape = list(grad_output.shape)
                    for ax in axis:
                        if not keepdims:
                            expand_shape.insert(ax, 1)
                else:
                    expand_shape = [1] * self.ndim
                
                grad = np.broadcast_to(grad_output.data.reshape(expand_shape), self.shape)
                self.backward(Tensor(grad))
            
            result.grad_fn = type('SumBackward', (), {'backward': sum_backward})
        
        return result
    
    def mean(self, axis=None, keepdims=False):
        """均值运算"""
        result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def mean_backward(grad_output):
                # 计算元素数量用于梯度缩放
                if axis is None:
                    num_elements = self.data.size
                else:
                    if not isinstance(axis, tuple):
                        axis = (axis,)
                    num_elements = 1
                    for ax in axis:
                        num_elements *= self.data.shape[ax]
                
                # 广播梯度到原始形状
                if axis is not None:
                    if not isinstance(axis, tuple):
                        axis = (axis,)
                    expand_shape = list(grad_output.shape)
                    for ax in axis:
                        if not keepdims:
                            expand_shape.insert(ax, 1)
                else:
                    expand_shape = [1] * self.ndim
                
                grad = np.broadcast_to(grad_output.data.reshape(expand_shape), self.shape) / num_elements
                self.backward(Tensor(grad))
            
            result.grad_fn = type('MeanBackward', (), {'backward': mean_backward})
        
        return result

# 辅助函数
def tensor(data, dtype=None, requires_grad=False):
    """创建张量的工厂函数"""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)

def zeros(shape, dtype=np.float32, requires_grad=False):
    """创建全零张量"""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def ones(shape, dtype=np.float32, requires_grad=False):
    """创建全一张量"""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

def randn(shape, dtype=np.float32, requires_grad=False):
    """创建标准正态分布的随机张量"""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)