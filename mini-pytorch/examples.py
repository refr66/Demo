from tensor import tensor, zeros, ones, randn
import nn
import optim
import functional as F
import numpy as np

# 示例1：基本张量操作
def example_tensor_operations():
    print("=" * 50)
    print("示例1：基本张量操作")
    print("=" * 50)
    
    # 创建张量
    x = tensor([1.0, 2.0, 3.0])
    y = tensor([4.0, 5.0, 6.0])
    print(f"x = {x}")
    print(f"y = {y}")
    
    # 基本运算
    z = x + y
    print(f"x + y = {z}")
    
    z = x * y
    print(f"x * y = {z}")
    
    # 自动微分
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()
    print(f"x.grad = {x.grad}")  # 应该是 [2, 2, 2]
    
    # 矩阵乘法
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    print(f"a @ b = {c}")

# 示例2：简单神经网络
def example_simple_neural_network():
    print("=" * 50)
    print("示例2：简单神经网络")
    print("=" * 50)
    
    # 创建一个简单的神经网络
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # 准备数据
    x = randn((32, 10), requires_grad=True)
    y = randn((32, 1))
    
    # 前向传播
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(output, y)
    print(f"Loss: {loss}")
    
    # 反向传播
    loss.backward()
    
    # 使用优化器更新参数
    optimizer = optim.SGD(model.parameters, lr=0.01)
    optimizer.step()
    
    # 清空梯度
    optimizer.zero_grad()

# 示例3：训练简单模型
def example_training_model():
    print("=" * 50)
    print("示例3：训练简单模型")
    print("=" * 50)
    
    # 创建数据集（线性回归问题）
    np.random.seed(42)
    X = np.random.rand(100, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # y = 2x + 1 + 噪声
    
    # 转换为张量
    X_tensor = tensor(X, requires_grad=True)
    y_tensor = tensor(y)
    
    # 创建模型
    model = nn.Linear(1, 1)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters, lr=0.1)
    
    # 训练循环
    epochs = 100
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.data:.4f}')
    
    # 打印训练后的参数
    print(f"训练后的权重: {model.weight.data}")
    print(f"训练后的偏置: {model.bias.data}")

# 示例4：使用函数式API
def example_functional_api():
    print("=" * 50)
    print("示例4：使用函数式API")
    print("=" * 50)
    
    # 创建张量
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 使用函数式API
    y = F.relu(x)
    z = F.sum(y)
    
    # 反向传播
    z.backward()
    
    print(f"x = {x}")
    print(f"F.relu(x) = {y}")
    print(f"F.sum(F.relu(x)) = {z}")
    print(f"x.grad = {x.grad}")

# 示例5：多分类问题
def example_multi_class_classification():
    print("=" * 50)
    print("示例5：多分类问题")
    print("=" * 50)
    
    # 创建数据集（三分类问题）
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # 转换为张量
    X_tensor = tensor(X, requires_grad=True)
    y_tensor = tensor(y)
    
    # 创建模型
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters, lr=0.01)
    
    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.data:.4f}')

# 运行所有示例
if __name__ == "__main__":
    example_tensor_operations()
    example_simple_neural_network()
    example_training_model()
    example_functional_api()
    example_multi_class_classification()