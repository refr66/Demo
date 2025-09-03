from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义扩展模块
setup(
    name='flash_attention_cuda_ext',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            'flash_attention_cuda_ext',
            sources=[
                os.path.join(current_dir, 'flash_attention_cuda_kernel.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-arch=sm_70',  # 适配安培架构及更高版本GPU
                    '-maxrregcount=256',  # 控制寄存器使用数量
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    description='CUDA implementation of Flash Attention',
    author='AI System Developer',
    author_email='aisys@example.com',
    license='MIT',
    keywords=['flash attention', 'cuda', 'pytorch', 'transformer'],
    url='',
)