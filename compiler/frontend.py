import torch
import torch.fx
from ir import IRGraph, TensorNode, ParameterNode, AddNode, MatmulNode, ReLUNode, SigmoidNode, TanhNode

class PyTorchFrontend:
    """PyTorch前端：将PyTorch计算图转换为IR"""
    def __init__(self):
        self.node_id = 0  # 用于生成唯一的节点名称
        self.ir_graph = IRGraph()  # IR图
        self.torch_to_ir_map = {}  # PyTorch节点到IR节点的映射
    
    def _get_next_node_name(self):
        """生成唯一的节点名称"""
        name = f"node_{self.node_id}"
        self.node_id += 1
        return name
    
    def _create_tensor_node(self, torch_tensor, name=None):
        """创建张量节点"""
        name = name or self._get_next_node_name()
        shape = list(torch_tensor.shape) if hasattr(torch_tensor, 'shape') else None
        dtype = str(torch_tensor.dtype) if hasattr(torch_tensor, 'dtype') else None
        tensor_node = TensorNode(name, shape=shape, dtype=dtype)
        self.ir_graph.add_node(tensor_node)
        self.torch_to_ir_map[id(torch_tensor)] = tensor_node
        return tensor_node
    
    def _create_parameter_node(self, torch_param, name):
        """创建参数节点"""
        param_node = ParameterNode(name, value=torch_param.data, requires_grad=torch_param.requires_grad)
        self.ir_graph.add_node(param_node)
        self.torch_to_ir_map[id(torch_param)] = param_node
        return param_node
    
    def _convert_call_function(self, target, args, kwargs):
        """转换call_function类型的节点"""
        # 检查args和kwargs中的张量
        ir_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) or id(arg) in self.torch_to_ir_map:
                ir_args.append(self.torch_to_ir_map.get(id(arg), self._create_tensor_node(arg)))
            else:
                ir_args.append(arg)
        
        # 处理不同类型的操作
        if target is torch.add or target is torch.Tensor.add:
            # 创建加法节点
            add_node = AddNode(self._get_next_node_name())
            for arg in ir_args:
                if isinstance(arg, (TensorNode, ParameterNode)):
                    add_node.add_input(arg)
            self.ir_graph.add_node(add_node)
            return add_node
        elif target is torch.matmul or target is torch.Tensor.matmul or target is torch.Tensor.__matmul__:
            # 创建矩阵乘法节点
            matmul_node = MatmulNode(self._get_next_node_name())
            for arg in ir_args:
                if isinstance(arg, (TensorNode, ParameterNode)):
                    matmul_node.add_input(arg)
            self.ir_graph.add_node(matmul_node)
            return matmul_node
        elif target is torch.relu or target is torch.Tensor.relu:
            # 创建ReLU节点
            relu_node = ReLUNode(self._get_next_node_name())
            for arg in ir_args:
                if isinstance(arg, (TensorNode, ParameterNode)):
                    relu_node.add_input(arg)
            self.ir_graph.add_node(relu_node)
            return relu_node
        elif target is torch.sigmoid or target is torch.Tensor.sigmoid:
            # 创建Sigmoid节点
            sigmoid_node = SigmoidNode(self._get_next_node_name())
            for arg in ir_args:
                if isinstance(arg, (TensorNode, ParameterNode)):
                    sigmoid_node.add_input(arg)
            self.ir_graph.add_node(sigmoid_node)
            return sigmoid_node
        elif target is torch.tanh or target is torch.Tensor.tanh:
            # 创建Tanh节点
            tanh_node = TanhNode(self._get_next_node_name())
            for arg in ir_args:
                if isinstance(arg, (TensorNode, ParameterNode)):
                    tanh_node.add_input(arg)
            self.ir_graph.add_node(tanh_node)
            return tanh_node
        else:
            # 对于不支持的操作，创建一个通用的张量节点
            print(f"Warning: Unsupported operation {target}")
            return self._create_tensor_node(torch.zeros(1))  # 占位符
    
    def _convert_module(self, module, inputs):
        """转换Module类型的节点"""
        # 处理模块的参数
        for name, param in module.named_parameters():
            self._create_parameter_node(param, f"{module.__class__.__name__}_{name}")
        
        # 对于简单的线性层，我们可以手动转换
        if isinstance(module, torch.nn.Linear):
            # 捕获权重和偏置参数
            weight = self.torch_to_ir_map.get(id(module.weight))
            bias = self.torch_to_ir_map.get(id(module.bias)) if module.bias is not None else None
            
            # 处理输入
            input_tensor = self.torch_to_ir_map.get(id(inputs[0]))
            
            # 创建矩阵乘法节点
            matmul_node = MatmulNode(self._get_next_node_name())
            matmul_node.add_input(input_tensor)
            matmul_node.add_input(weight)
            self.ir_graph.add_node(matmul_node)
            
            # 如果有偏置，添加加法节点
            if bias is not None:
                add_node = AddNode(self._get_next_node_name())
                add_node.add_input(matmul_node)
                add_node.add_input(bias)
                self.ir_graph.add_node(add_node)
                return add_node
            
            return matmul_node
        
        # 对于ReLU、Sigmoid、Tanh等激活函数层
        elif isinstance(module, torch.nn.ReLU):
            relu_node = ReLUNode(self._get_next_node_name())
            input_tensor = self.torch_to_ir_map.get(id(inputs[0]))
            relu_node.add_input(input_tensor)
            self.ir_graph.add_node(relu_node)
            return relu_node
        elif isinstance(module, torch.nn.Sigmoid):
            sigmoid_node = SigmoidNode(self._get_next_node_name())
            input_tensor = self.torch_to_ir_map.get(id(inputs[0]))
            sigmoid_node.add_input(input_tensor)
            self.ir_graph.add_node(sigmoid_node)
            return sigmoid_node
        elif isinstance(module, torch.nn.Tanh):
            tanh_node = TanhNode(self._get_next_node_name())
            input_tensor = self.torch_to_ir_map.get(id(inputs[0]))
            tanh_node.add_input(input_tensor)
            self.ir_graph.add_node(tanh_node)
            return tanh_node
        
        # 对于不支持的模块，创建一个通用的张量节点
        print(f"Warning: Unsupported module {module.__class__.__name__}")
        return self._create_tensor_node(torch.zeros(1))  # 占位符
    
    def convert(self, fx_module, input_tensors):
        """将FX模块转换为IR图"""
        # 记录输入张量
        for i, tensor in enumerate(input_tensors):
            self._create_tensor_node(tensor, name=f"input_{i}")
        
        # 遍历FX图中的节点
        for node in fx_module.graph.nodes:
            if node.op == 'placeholder':
                # 占位符节点已经作为输入处理
                pass
            elif node.op == 'get_attr':
                # 获取模块属性（如权重、偏置）
                attr = fx_module.getattr(node.target)
                if isinstance(attr, torch.nn.Parameter):
                    self._create_parameter_node(attr, node.name)
            elif node.op == 'call_function':
                # 处理函数调用（如torch.add, torch.matmul等）
                result = self._convert_call_function(node.target, node.args, node.kwargs)
                # 如果结果是IR节点，保存映射
                if isinstance(result, TensorNode):
                    self.torch_to_ir_map[id(node)] = result
            elif node.op == 'call_module':
                # 处理模块调用（如nn.Linear, nn.ReLU等）
                module = fx_module.get_submodule(node.target)
                # 准备输入参数
                inputs = []
                for arg in node.args:
                    if isinstance(arg, torch.Tensor) or id(arg) in self.torch_to_ir_map:
                        inputs.append(self.torch_to_ir_map.get(id(arg), arg))
                    else:
                        inputs.append(arg)
                # 转换模块
                result = self._convert_module(module, inputs)
                # 保存映射
                if isinstance(result, TensorNode):
                    self.torch_to_ir_map[id(node)] = result
            elif node.op == 'output':
                # 设置输出节点
                output_nodes = []
                for output in node.args:
                    if isinstance(output, torch.Tensor) or id(output) in self.torch_to_ir_map:
                        output_nodes.append(self.torch_to_ir_map.get(id(output)))
                self.ir_graph.set_outputs(output_nodes)
        
        return self.ir_graph
    
    def capture_and_convert(self, model, input_tensors):
        """捕获模型的计算图并转换为IR"""
        # 使用torch.fx捕获模型的计算图
        fx_module = torch.fx.symbolic_trace(model)
        # 转换为IR图
        return self.convert(fx_module, input_tensors)

# 快捷函数
def torch_to_ir(model, input_tensors):
    """将PyTorch模型转换为IR图"""
    frontend = PyTorchFrontend()
    return frontend.capture_and_convert(model, input_tensors)