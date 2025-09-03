from tensor import Tensor

class Optimizer:
    """优化器基类"""
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """执行单步优化，需要被子类重写"""
        raise NotImplementedError
    
    def zero_grad(self):
        """清空所有参数的梯度"""
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
            elif isinstance(param, list):
                for p in param:
                    if hasattr(p, 'zero_grad'):
                        p.zero_grad()

class SGD(Optimizer):
    """随机梯度下降优化器"""
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def step(self):
        """执行单步SGD优化"""
        for param in self.parameters:
            # 处理模块中的参数
            if hasattr(param, 'parameters'):
                params = param.parameters
            else:
                params = [param] if not isinstance(param, list) else param
            
            for p in params:
                if p.grad is None:
                    continue
                
                # 权重衰减
                if self.weight_decay > 0:
                    p.data += -self.weight_decay * self.lr * p.data
                
                # 动量
                if self.momentum > 0:
                    if id(p) not in self.velocities:
                        self.velocities[id(p)] = p.grad.copy()
                    else:
                        self.velocities[id(p)] = self.momentum * self.velocities[id(p)] + p.grad
                    
                    # 更新参数
                    p.data += -self.lr * self.velocities[id(p)]
                else:
                    # 标准SGD更新
                    p.data += -self.lr * p.grad

class Adam(Optimizer):
    """Adam优化器"""
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self):
        """执行单步Adam优化"""
        self.t += 1
        
        for param in self.parameters:
            # 处理模块中的参数
            if hasattr(param, 'parameters'):
                params = param.parameters
            else:
                params = [param] if not isinstance(param, list) else param
            
            for p in params:
                if p.grad is None:
                    continue
                
                # 权重衰减
                if self.weight_decay > 0:
                    p.data += -self.weight_decay * self.lr * p.data
                
                # 初始化一阶和二阶矩估计
                if id(p) not in self.m:
                    self.m[id(p)] = np.zeros_like(p.data)
                    self.v[id(p)] = np.zeros_like(p.data)
                
                # 更新一阶和二阶矩估计
                self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * p.grad
                self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (p.grad ** 2)
                
                # 偏差校正
                m_hat = self.m[id(p)] / (1 - self.beta1 ** self.t)
                v_hat = self.v[id(p)] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                p.data += -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 快捷访问
def sgd(parameters, lr=0.01, momentum=0, weight_decay=0):
    return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

def adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    return Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)