# File: setup.py

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the extension module
ext_modules = [
    CUDAExtension(
        name='int4_matmul_cuda',
        sources=[
            os.path.join(current_dir, 'int4_cuda.cpp'),
            os.path.join(current_dir, 'Quantization', 'GPTQ', 'packing', 'int4_pack.cu'),
            os.path.join(current_dir, 'Quantization', 'GPTQ', 'matmul', 'int4_matmul.cu')
        ],
        include_dirs=[
            os.path.join(current_dir, 'Quantization'),
            os.path.join(current_dir, 'Quantization', 'GPTQ')
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++14']
        }
    ),
]

# Setup
setup(
    name='int4_matmul',
    version='0.1',
    author='Ken Ding + Claude 3.5 Sonnet',
    author_email='n/a',
    description='INT4 MatMul CUDA Extension',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
)