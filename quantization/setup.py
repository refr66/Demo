from setuptools import setup, find_packages
import os

# 读取requirements.txt文件中的依赖
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
    requirements = [line.strip() for line in f if line.strip()]

# 读取README.md文件作为long_description
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ptq-toolkit',  # 包名称
    version='0.1.0',    # 版本号
    description='A simple Post-Training Quantization (PTQ) toolkit for deep learning models',  # 简短描述
    long_description=long_description,  # 详细描述
    long_description_content_type='text/markdown',  # 详细描述格式
    author='AI Researcher',  # 作者
    author_email='ai.researcher@example.com',  # 作者邮箱
    url='https://github.com/example/ptq-toolkit',  # 项目URL（示例）
    packages=find_packages(),  # 自动发现包
    install_requires=requirements,  # 依赖项
    classifiers=[  # 包分类
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='quantization, deep learning, model compression, int8, ptq',  # 关键词
    python_requires='>=3.8',  # Python版本要求
    license='MIT',  # 许可证
)

# 注意：这是一个示例setup.py文件，实际使用时需要根据项目情况修改相关信息，如作者、URL、许可证等。