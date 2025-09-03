from setuptools import setup, find_packages

setup(
    name="mini-pytorch",
    version="0.1.0",
    description="A simplified implementation of PyTorch core functionalities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)