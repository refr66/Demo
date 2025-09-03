import setuptools
import os

# 读取requirements.txt文件获取依赖项
def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# 读取README.md文件作为长描述
def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

# 设置包的元数据
setuptools.setup(
    name="distributed-parallel-training",
    version="0.1.0",
    author="AI Sys Team",
    author_email="aisys@example.com",
    description="分布式训练并行策略实现（数据并行、流水线并行、张量并行）",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/distributed-parallel-training",
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    keywords=[
        "distributed training",
        "parallel computing",
        "data parallelism",
        "pipeline parallelism",
        "tensor parallelism",
        "deep learning",
        "PyTorch"
    ],
    # 可选的额外依赖项
    extras_require={
        "dev": [
            "pytest>=7.0",
            "flake8>=4.0",
            "black>=22.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    # 包含的数据文件
    package_data={
        "parallel": ["requirements.txt", "README.md"],
    },
    # 指示包是纯Python包
    zip_safe=False,
)