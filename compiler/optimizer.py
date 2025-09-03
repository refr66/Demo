from ir import IRGraph, AddNode, ReLUNode, MatmulNode, IRNode, FusedAddReluNode, FusedMatmulAddNode

class GraphOptimizer:
    """图优化器：实现各种图优化pass"""
    def __init__(self, ir_graph):
        self.ir_graph = ir_graph
        self.passes = [
            self.fuse_add_relu,  # 融合add和relu
            self.fuse_matmul_add,  # 融合matmul和add
            self.remove_unused_nodes,  # 移除未使用的节点
            self.constant_folding,  # 常量折叠
            self.common_subexpression_elimination,  # 公共子表达式消除
        ]
    
    def optimize(self):
        """运行所有优化pass"""
        print("Running graph optimizations...")
        for i, pass_func in enumerate(self.passes):
            print(f"Running pass {i+1}/{len(self.passes)}: {pass_func.__name__}")
            self.ir_graph = pass_func(self.ir_graph)
        print("Graph optimizations completed.")
        return self.ir_graph
    
    def fuse_add_relu(self, graph):
        """融合add和relu操作"""
        # 按照拓扑顺序遍历图
        for node in graph.topological_order():
            # 查找add节点后面跟着relu节点的情况
            if isinstance(node, AddNode):
                for next_node in list(node.outputs):  # 使用列表副本以避免在遍历时修改集合
                    if isinstance(next_node, ReLUNode) and len(next_node.inputs) == 1 and next_node.inputs[0] == node:
                        # 创建融合后的节点
                        fused_node = FusedAddReluNode(
                            name=f"fused_{node.name}_{next_node.name}",
                            inputs=list(node.inputs),
                            shape=next_node.shape if hasattr(next_node, 'shape') else None
                        )
                        
                        # 更新后续节点的输入
                        for output_node in list(next_node.outputs):
                            # 移除对next_node的引用
                            if next_node in output_node.inputs:
                                output_node.inputs.remove(next_node)
                                # 添加对fused_node的引用
                                output_node.add_input(fused_node)
                        
                        # 将融合后的节点添加到图中
                        graph.add_node(fused_node)
                        
                        # 从图中移除原始节点
                        if node in graph.nodes:
                            graph.nodes.remove(node)
                        if next_node in graph.nodes:
                            graph.nodes.remove(next_node)
                        
                        # 更新输出节点列表
                        if next_node in graph.outputs:
                            graph.outputs.remove(next_node)
                            graph.outputs.append(fused_node)
                        
                        print(f"Fused Add({node.name}) and ReLU({next_node.name}) into FusedAddRelu({fused_node.name})")
        
        return graph
    
    def fuse_matmul_add(self, graph):
        """融合matmul和add操作"""
        # 按照拓扑顺序遍历图
        for node in graph.topological_order():
            # 查找matmul节点后面跟着add节点的情况
            if isinstance(node, MatmulNode):
                for next_node in list(node.outputs):
                    if isinstance(next_node, AddNode) and len(next_node.inputs) == 2 and next_node.inputs[0] == node:
                        # 检查第二个输入是否是常量或参数
                        second_input = next_node.inputs[1]
                        if hasattr(second_input, 'value') or second_input.op_type == 'parameter':
                            # 创建融合后的节点
                            fused_node = FusedMatmulAddNode(
                                name=f"fused_{node.name}_{next_node.name}",
                                inputs=list(node.inputs) + [second_input],
                                shape=next_node.shape if hasattr(next_node, 'shape') else None
                            )
                            
                            # 更新后续节点的输入
                            for output_node in list(next_node.outputs):
                                if next_node in output_node.inputs:
                                    output_node.inputs.remove(next_node)
                                    output_node.add_input(fused_node)
                            
                            # 将融合后的节点添加到图中
                            graph.add_node(fused_node)
                            
                            # 从图中移除原始节点
                            if node in graph.nodes:
                                graph.nodes.remove(node)
                            if next_node in graph.nodes:
                                graph.nodes.remove(next_node)
                            
                            # 更新输出节点列表
                            if next_node in graph.outputs:
                                graph.outputs.remove(next_node)
                                graph.outputs.append(fused_node)
                            
                            print(f"Fused Matmul({node.name}) and Add({next_node.name}) into FusedMatmulAdd({fused_node.name})")
        
        return graph
    
    def remove_unused_nodes(self, graph):
        """移除未使用的节点（死代码消除）"""
        # 找出所有可达的节点
        reachable = set()
        
        def mark_reachable(node):
            if node in reachable:
                return
            reachable.add(node)
            for input_node in node.inputs:
                mark_reachable(input_node)
        
        # 从输出节点开始标记可达节点
        for output_node in graph.outputs:
            mark_reachable(output_node)
        
        # 移除不可达的节点
        nodes_to_remove = [node for node in graph.nodes if node not in reachable]
        for node in nodes_to_remove:
            graph.nodes.remove(node)
        
        # 更新输入节点列表
        graph.inputs = [node for node in graph.inputs if node in reachable]
        
        print(f"Removed {len(nodes_to_remove)} unused nodes")
        return graph
    
    def constant_folding(self, graph):
        """常量折叠优化"""
        # 遍历图中的所有节点
        for node in list(graph.nodes):
            # 检查节点是否是可折叠的操作，并且所有输入都是常量
            if self._is_foldable(node) and self._all_inputs_are_constants(node):
                # 尝试执行常量折叠
                try:
                    folded_value = self._compute_constant_value(node)
                    if folded_value is not None:
                        # 创建一个新的参数节点作为常量结果
                        from ir import ParameterNode
                        constant_node = ParameterNode(
                            name=f"folded_{node.name}",
                            value=folded_value,
                            shape=node.shape if hasattr(node, 'shape') else None
                        )
                        
                        # 更新后续节点的输入
                        for output_node in list(node.outputs):
                            if node in output_node.inputs:
                                output_node.inputs.remove(node)
                                output_node.add_input(constant_node)
                        
                        # 将常量节点添加到图中
                        graph.add_node(constant_node)
                        
                        # 从图中移除原始节点
                        if node in graph.nodes:
                            graph.nodes.remove(node)
                        
                        # 更新输出节点列表
                        if node in graph.outputs:
                            graph.outputs.remove(node)
                            graph.outputs.append(constant_node)
                        
                        print(f"Folded {node.op_type} node {node.name} into constant parameter")
                except Exception as e:
                    print(f"Failed to fold {node.name}: {e}")
        
        return graph
    
    def _is_foldable(self, node):
        """检查节点是否是可折叠的操作"""
        # 支持的可折叠操作
        foldable_ops = ["add", "matmul", "relu", "sigmoid", "tanh"]
        return node.op_type in foldable_ops
    
    def _all_inputs_are_constants(self, node):
        """检查节点的所有输入是否都是常量"""
        for input_node in node.inputs:
            # 假设parameter节点都是常量
            if input_node.op_type not in ["parameter"]:
                return False
        return True
    
    def _compute_constant_value(self, node):
        """计算常量节点的值"""
        # 从节点获取输入值
        input_values = []
        for input_node in node.inputs:
            if hasattr(input_node, 'value') and input_node.value is not None:
                input_values.append(input_node.value)
            else:
                return None
        
        # 根据操作类型计算结果
        import numpy as np
        
        if node.op_type == "add" and len(input_values) >= 2:
            return np.add(input_values[0], input_values[1])
        elif node.op_type == "matmul" and len(input_values) >= 2:
            return np.matmul(input_values[0], input_values[1])
        elif node.op_type == "relu" and len(input_values) >= 1:
            return np.maximum(0, input_values[0])
        elif node.op_type == "sigmoid" and len(input_values) >= 1:
            return 1.0 / (1.0 + np.exp(-input_values[0]))
        elif node.op_type == "tanh" and len(input_values) >= 1:
            return np.tanh(input_values[0])
        
        return None
    
    def common_subexpression_elimination(self, graph):
        """公共子表达式消除优化"""
        # 创建一个哈希表来存储已经计算过的表达式
        expression_map = {}
        
        # 按照拓扑顺序遍历图
        for node in graph.topological_order():
            # 跳过输入节点和参数节点
            if node.op_type in ["tensor", "parameter"]:
                continue
            
            # 生成表达式的哈希键
            expr_key = self._generate_expression_key(node)
            
            # 检查是否已经存在相同的表达式
            if expr_key in expression_map:
                # 存在相同的表达式，使用已有的节点替换当前节点
                existing_node = expression_map[expr_key]
                
                # 更新后续节点的输入
                for output_node in list(node.outputs):
                    if node in output_node.inputs:
                        output_node.inputs.remove(node)
                        output_node.add_input(existing_node)
                
                # 从图中移除当前节点
                if node in graph.nodes:
                    graph.nodes.remove(node)
                
                # 更新输出节点列表
                if node in graph.outputs:
                    graph.outputs.remove(node)
                    if existing_node not in graph.outputs:
                        graph.outputs.append(existing_node)
                
                print(f"Eliminated common subexpression: {node.name} replaced with {existing_node.name}")
            else:
                # 将当前表达式添加到哈希表中
                expression_map[expr_key] = node
        
        return graph
    
    def _generate_expression_key(self, node):
        """生成表达式的哈希键"""
        # 表达式的哈希键由操作类型和输入节点的名称组成
        input_names = tuple(sorted([input_node.name for input_node in node.inputs]))
        return (node.op_type, input_names)
    
    def add_custom_pass(self, pass_func):
        """添加自定义优化pass"""
        self.passes.append(pass_func)

class PassManager:
    """优化Pass管理器，负责运行一系列优化Pass"""
    def __init__(self, passes=None):
        self.passes = passes or []
        
    def add_pass(self, opt_pass):
        """添加一个优化Pass"""
        self.passes.append(opt_pass)
        
    def run(self, ir_graph):
        """运行所有优化Pass"""
        for opt_pass in self.passes:
            optimizer = opt_pass(ir_graph)
            optimizer.optimize()
        return ir_graph

# 快捷函数
def create_default_optimizers():
    """创建默认的优化器Pass列表"""
    return PassManager([
        GraphOptimizer
    ])

def optimize_ir(ir_graph, passes=None):
    """优化IR图"""
    if passes is None:
        passes = GraphOptimizer(ir_graph)
        return passes.optimize()
    else:
        return passes.run(ir_graph)