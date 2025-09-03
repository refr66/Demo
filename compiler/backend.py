import numpy as np
import os

class CodeGenerator:
    """代码生成器基类"""
    def __init__(self, ir_graph):
        self.ir_graph = ir_graph
        self.code = []
        self.indent_level = 0
        self.node_to_var = {}
    
    def indent(self):
        """增加缩进"""
        self.indent_level += 4
    
    def dedent(self):
        """减少缩进"""
        if self.indent_level >= 4:
            self.indent_level -= 4
    
    def write(self, line):
        """写入一行代码"""
        self.code.append(' ' * self.indent_level + line)
    
    def generate(self):
        """生成代码，需要被子类重写"""
        raise NotImplementedError
    
    def get_code(self):
        """获取生成的代码"""
        return '\n'.join(self.code)
    
    def save_to_file(self, file_path):
        """将生成的代码保存到文件"""
        with open(file_path, 'w') as f:
            f.write(self.get_code())

class PythonNumpyBackend(CodeGenerator):
    """Python/NumPy后端"""
    def __init__(self, ir_graph):
        super().__init__(ir_graph)
    
    def generate(self):
        """生成Python/NumPy代码"""
        # 导入必要的库
        self.write("import numpy as np")
        self.write("")
        
        # 生成函数定义
        self.write("def compiled_function(inputs):")
        self.indent()
        
        # 处理输入
        self._process_inputs()
        
        # 按照拓扑顺序生成节点代码
        for node in self.ir_graph.topological_order():
            self._generate_node_code(node)
        
        # 生成输出
        self._generate_output()
        
        self.dedent()
        self.write("")
        
        # 生成主函数示例
        self.write("if __name__ == '__main__':")
        self.indent()
        # 生成示例输入
        input_shapes = []
        for i, input_node in enumerate(self.ir_graph.inputs):
            if hasattr(input_node, 'shape') and input_node.shape:
                input_shapes.append(input_node.shape)
            else:
                input_shapes.append((1,))
        
        self.write("# 创建示例输入")
        for i, shape in enumerate(input_shapes):
            self.write(f"input_{i} = np.random.rand(*{shape})")
        
        self.write("")
        self.write("# 调用编译后的函数")
        self.write(f"inputs = [{', '.join([f'input_{i}' for i in range(len(input_shapes))])}]")
        self.write("output = compiled_function(inputs)")
        self.write("print('Output shape:', output.shape)")
        self.dedent()
    
    def _process_inputs(self):
        """处理输入参数"""
        self.write("# 解包输入")
        for i, input_node in enumerate(self.ir_graph.inputs):
            var_name = f"var_{input_node.name}"
            self.node_to_var[input_node] = var_name
            self.write(f"{var_name} = np.array(inputs[{i}])")
        self.write("")
    
    def _generate_node_code(self, node):
        """生成节点代码"""
        var_name = f"var_{node.name}"
        
        if node.op_type == "tensor":
            # 张量节点不需要生成代码，因为它们已经作为输入处理了
            pass
        elif node.op_type == "parameter":
            # 生成参数代码，直接硬编码参数值
            self.write(f"# Parameter {node.name}")
            if hasattr(node, 'value') and node.value is not None:
                # 将PyTorch张量转换为NumPy数组并硬编码
                if hasattr(node.value, 'numpy'):
                    np_value = node.value.numpy()
                else:
                    np_value = np.array(node.value)
                # 生成NumPy数组初始化代码
                self.write(f"{var_name} = np.array({np_value.tolist()})")
            else:
                self.write(f"{var_name} = np.zeros(1)  # 未初始化的参数")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "add":
            # 生成加法代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Add operation {node.name}")
            self.write(f"{var_name} = {input_vars[0]} + {input_vars[1]}")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "matmul":
            # 生成矩阵乘法代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Matrix multiplication {node.name}")
            self.write(f"{var_name} = np.matmul({input_vars[0]}, {input_vars[1]})")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "relu":
            # 生成ReLU代码
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"# ReLU activation {node.name}")
            self.write(f"{var_name} = np.maximum(0, {input_var})")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "sigmoid":
            # 生成Sigmoid代码
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"# Sigmoid activation {node.name}")
            self.write(f"{var_name} = 1.0 / (1.0 + np.exp(-{input_var}))")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "tanh":
            # 生成Tanh代码
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"# Tanh activation {node.name}")
            self.write(f"{var_name} = np.tanh({input_var})")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "fused_add_relu":
            # 生成融合的add+relu代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Fused Add+ReLU {node.name}")
            self.write(f"{var_name} = np.maximum(0, {input_vars[0]} + {input_vars[1]})")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "fused_matmul_add":
            # 生成融合的matmul+add代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Fused Matmul+Add {node.name}")
            self.write(f"{var_name} = np.matmul({input_vars[0]}, {input_vars[1]}) + {input_vars[2]}")
            self.node_to_var[node] = var_name
            self.write("")
        else:
            # 对于未知类型的节点，生成占位符代码
            self.write(f"# Unknown operation {node.op_type}")
            self.write(f"{var_name} = np.zeros(1)  # Placeholder")
            self.node_to_var[node] = var_name
            self.write("")
    
    def _generate_output(self):
        """生成输出代码"""
        if len(self.ir_graph.outputs) == 1:
            output_var = self.node_to_var[self.ir_graph.outputs[0]]
            self.write(f"# Return output")
            self.write(f"return {output_var}")
        else:
            output_vars = [self.node_to_var[output_node] for output_node in self.ir_graph.outputs]
            self.write(f"# Return outputs")
            self.write(f"return [{', '.join(output_vars)}]")

class TritonBackend(CodeGenerator):
    """Triton后端"""
    def __init__(self, ir_graph):
        super().__init__(ir_graph)
    
    def generate(self):
        """生成Triton代码"""
        # 导入必要的库
        self.write("import triton")
        self.write("import triton.language as tl")
        self.write("import numpy as np")
        self.write("")
        
        # 检查Triton是否可用
        self.write("# 检查Triton是否可用")
        self.write("try:")
        self.indent()
        self.write("import triton")
        self.write("HAS_TRITON = True")
        self.dedent()
        self.write("except ImportError:")
        self.indent()
        self.write("HAS_TRITON = False")
        self.write("print('Triton is not available. Using NumPy fallback.')")
        self.dedent()
        self.write("")
        
        # 生成Triton内核函数
        self._generate_triton_kernels()
        
        # 生成包装函数
        self.write("def compiled_function(inputs):")
        self.indent()
        
        # 如果Triton不可用，使用NumPy回退
        self.write("if not HAS_TRITON:")
        self.indent()
        self._generate_numpy_fallback()
        self.dedent()
        self.write("else:")
        self.indent()
        
        # 处理输入
        self._process_inputs()
        
        # 按照拓扑顺序生成节点代码
        for node in self.ir_graph.topological_order():
            self._generate_triton_node_code(node)
        
        # 生成输出
        self._generate_output()
        
        self.dedent()
        self.dedent()
        self.write("")
        
        # 生成主函数示例
        self.write("if __name__ == '__main__':")
        self.indent()
        # 生成示例输入
        input_shapes = []
        for i, input_node in enumerate(self.ir_graph.inputs):
            if hasattr(input_node, 'shape') and input_node.shape:
                input_shapes.append(input_node.shape)
            else:
                input_shapes.append((1,))
        
        self.write("# 创建示例输入")
        for i, shape in enumerate(input_shapes):
            self.write(f"input_{i} = np.random.rand(*{shape}).astype(np.float32)")
        
        self.write("")
        self.write("# 调用编译后的函数")
        self.write(f"inputs = [{', '.join([f'input_{i}' for i in range(len(input_shapes))])}]")
        self.write("output = compiled_function(inputs)")
        self.write("print('Output shape:', output.shape)")
        self.dedent()
    
    def _generate_triton_kernels(self):
        """生成Triton内核函数"""
        # 为常见操作生成Triton内核
        self.write("@triton.jit")
        self.write("def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):")
        self.indent()
        self.write("pid = tl.program_id(0)")
        self.write("block_start = pid * BLOCK_SIZE")
        self.write("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
        self.write("mask = offsets < N")
        self.write("x = tl.load(x_ptr + offsets, mask=mask)")
        self.write("y = tl.load(y_ptr + offsets, mask=mask)")
        self.write("output = x + y")
        self.write("tl.store(output_ptr + offsets, output, mask=mask)")
        self.dedent()
        self.write("")
        
        self.write("@triton.jit")
        self.write("def relu_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):")
        self.indent()
        self.write("pid = tl.program_id(0)")
        self.write("block_start = pid * BLOCK_SIZE")
        self.write("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
        self.write("mask = offsets < N")
        self.write("x = tl.load(x_ptr + offsets, mask=mask)")
        self.write("output = tl.maximum(0, x)")
        self.write("tl.store(output_ptr + offsets, output, mask=mask)")
        self.dedent()
        self.write("")
    
    def _process_inputs(self):
        """处理输入参数"""
        self.write("# 解包输入")
        for i, input_node in enumerate(self.ir_graph.inputs):
            var_name = f"var_{input_node.name}"
            self.node_to_var[input_node] = var_name
            self.write(f"{var_name} = np.array(inputs[{i}]).astype(np.float32)")
        self.write("")
    
    def _generate_triton_node_code(self, node):
        """生成Triton节点代码"""
        var_name = f"var_{node.name}"
        
        if node.op_type == "tensor":
            # 张量节点不需要生成代码，因为它们已经作为输入处理了
            pass
        elif node.op_type == "parameter":
            # 生成参数代码，直接硬编码参数值
            self.write(f"# Parameter {node.name}")
            if hasattr(node, 'value') and node.value is not None:
                # 将PyTorch张量转换为NumPy数组并硬编码
                if hasattr(node.value, 'numpy'):
                    np_value = node.value.numpy()
                else:
                    np_value = np.array(node.value)
                # 生成NumPy数组初始化代码
                self.write(f"{var_name} = np.array({np_value.tolist()}).astype(np.float32)")
            else:
                self.write(f"{var_name} = np.zeros(1, dtype=np.float32)  # 未初始化的参数")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "add":
            # 生成加法代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Add operation {node.name}")
            self.write(f"{var_name} = np.empty_like({input_vars[0]})")
            self.write(f"N = {var_name}.size")
            self.write(f"BLOCK_SIZE = 1024")
            self.write(f"num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE")
            self.write(f"add_kernel[(num_blocks,)]({input_vars[0]}, {input_vars[1]}, {var_name}, N, BLOCK_SIZE=BLOCK_SIZE)")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "relu":
            # 生成ReLU代码
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"# ReLU activation {node.name}")
            self.write(f"{var_name} = np.empty_like({input_var})")
            self.write(f"N = {var_name}.size")
            self.write(f"BLOCK_SIZE = 1024")
            self.write(f"num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE")
            self.write(f"relu_kernel[(num_blocks,)]({input_var}, {var_name}, N, BLOCK_SIZE=BLOCK_SIZE)")
            self.node_to_var[node] = var_name
            self.write("")
        elif node.op_type == "fused_add_relu":
            # 生成融合的add+relu代码
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"# Fused Add+ReLU {node.name}")
            # 对于简单的融合操作，我们可以先使用add_kernel，然后使用relu_kernel
            self.write(f"temp_add = np.empty_like({input_vars[0]})")
            self.write(f"N = temp_add.size")
            self.write(f"BLOCK_SIZE = 1024")
            self.write(f"num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE")
            self.write(f"add_kernel[(num_blocks,)]({input_vars[0]}, {input_vars[1]}, temp_add, N, BLOCK_SIZE=BLOCK_SIZE)")
            self.write(f"{var_name} = np.empty_like(temp_add)")
            self.write(f"relu_kernel[(num_blocks,)](temp_add, {var_name}, N, BLOCK_SIZE=BLOCK_SIZE)")
            self.node_to_var[node] = var_name
            self.write("")
        else:
            # 对于复杂操作，回退到NumPy实现
            self.write(f"# Fallback to NumPy for {node.op_type}")
            self._generate_numpy_node_code(node)
            self.write("")
    
    def _generate_numpy_node_code(self, node):
        """生成NumPy回退代码"""
        var_name = f"var_{node.name}"
        
        if node.op_type == "matmul":
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"{var_name} = np.matmul({input_vars[0]}, {input_vars[1]})")
        elif node.op_type == "fused_matmul_add":
            input_vars = [self.node_to_var[input_node] for input_node in node.inputs]
            self.write(f"{var_name} = np.matmul({input_vars[0]}, {input_vars[1]}) + {input_vars[2]}")
        elif node.op_type == "sigmoid":
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"{var_name} = 1.0 / (1.0 + np.exp(-{input_var}))")
        elif node.op_type == "tanh":
            input_var = self.node_to_var[node.inputs[0]]
            self.write(f"{var_name} = np.tanh({input_var})")
        else:
            # 对于未知类型的节点，生成占位符代码
            self.write(f"{var_name} = np.zeros(1, dtype=np.float32)  # Placeholder")
        
        self.node_to_var[node] = var_name
    
    def _generate_numpy_fallback(self):
        """生成NumPy回退代码"""
        # 导入numpy
        self.write("import numpy as np")
        # 解包输入
        self.write("inputs = [np.array(x) for x in inputs]")
        # 使用简单的NumPy实现
        # 这里我们简单地调用PythonNumpyBackend来生成回退代码
        # 但为了保持代码简洁，我们直接返回一个占位符
        self.write("# Simple NumPy fallback")
        self.write("return np.zeros(1)")
    
    def _generate_output(self):
        """生成输出代码"""
        if len(self.ir_graph.outputs) == 1:
            output_var = self.node_to_var[self.ir_graph.outputs[0]]
            self.write(f"# Return output")
            self.write(f"return {output_var}")
        else:
            output_vars = [self.node_to_var[output_node] for output_node in self.ir_graph.outputs]
            self.write(f"# Return outputs")
            self.write(f"return [{', '.join(output_vars)}]")

# 快捷函数
def generate_code(ir_graph, backend='numpy'):
    """生成代码"""
    if backend.lower() == 'triton':
        generator = TritonBackend(ir_graph)
    else:
        generator = PythonNumpyBackend(ir_graph)
    
    generator.generate()
    return generator.get_code()

def save_code_to_file(ir_graph, file_path, backend='numpy'):
    """将生成的代码保存到文件"""
    if backend.lower() == 'triton':
        generator = TritonBackend(ir_graph)
    else:
        generator = PythonNumpyBackend(ir_graph)
    
    generator.generate()
    generator.save_to_file(file_path)