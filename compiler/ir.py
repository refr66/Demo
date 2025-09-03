class IRNode:
    """IR节点基类"""
    def __init__(self, name, op_type):
        self.name = name  # 节点名称
        self.op_type = op_type  # 操作类型
        self.inputs = []  # 输入节点列表
        self.outputs = []  # 输出节点列表
        self.attributes = {}  # 节点属性
        
    def add_input(self, node):
        """添加输入节点"""
        self.inputs.append(node)
        node.outputs.append(self)
        return self
    
    def __repr__(self):
        return f"{self.op_type}({self.name})"

class TensorNode(IRNode):
    """张量节点"""
    def __init__(self, name, shape=None, dtype=None):
        super().__init__(name, "tensor")
        self.shape = shape
        self.dtype = dtype
        self.attributes['shape'] = shape
        self.attributes['dtype'] = dtype

class ParameterNode(IRNode):
    """参数节点"""
    def __init__(self, name, value=None, requires_grad=True):
        super().__init__(name, "parameter")
        self.value = value  # 参数值
        self.requires_grad = requires_grad  # 是否需要梯度
        self.attributes['requires_grad'] = requires_grad

class AddNode(IRNode):
    """加法节点"""
    def __init__(self, name):
        super().__init__(name, "add")

class MatmulNode(IRNode):
    """矩阵乘法节点"""
    def __init__(self, name):
        super().__init__(name, "matmul")

class ReLUNode(IRNode):
    """ReLU激活函数节点"""
    def __init__(self, name):
        super().__init__(name, "relu")

class SigmoidNode(IRNode):
    """Sigmoid激活函数节点"""
    def __init__(self, name):
        super().__init__(name, "sigmoid")

class TanhNode(IRNode):
    """Tanh激活函数节点"""
    def __init__(self, name):
        super().__init__(name, "tanh")

class IRGraph:
    """中间表示计算图"""
    def __init__(self):
        self.nodes = []  # 所有节点
        self.inputs = []  # 输入节点
        self.outputs = []  # 输出节点
    
    def add_node(self, node):
        """添加节点到图中"""
        self.nodes.append(node)
        if isinstance(node, (TensorNode, ParameterNode)) and not node.inputs:
            self.inputs.append(node)
        return node
    
    def set_outputs(self, outputs):
        """设置图的输出节点"""
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
    
    def topological_order(self):
        """返回图的拓扑排序"""
        visited = set()
        result = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for next_node in node.outputs:
                # 确保所有输入节点都已访问
                all_inputs_visited = all(input_node in visited for input_node in next_node.inputs)
                if all_inputs_visited:
                    dfs(next_node)
            result.append(node)
        
        # 从输入节点开始DFS
        for input_node in self.inputs:
            dfs(input_node)
        
        return result
    
    def find_node_by_name(self, name):
        """通过名称查找节点"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def __repr__(self):
        return f"IRGraph with {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs"