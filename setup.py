from setuptools import setup, find_packages

setup(
    name='gptfast',
    version='0.1.0',
    author="MDK8888",
    description="Accelerate your transformer models by up to 6x. Native to Huggingface and PyTorch.",
    packages=find_packages(),
    install_requires=['torch==2.1.2', 'sympy==1.12', 'typing-extensions==4.9.0', 'networkx==3.2.1', 'jinja2==3.1.3', 'triton==2.1.0', 'fsspec==2023.10.0', 'filelock==3.13.1', 'MarkupSafe==2.1.4', 'mpmath==1.3.0', 'transformers==4.37.2', 'tqdm==4.66.1', 'pyyaml==5.4.1', 'safetensors==0.4.1', 'numpy==1.26.3', 'huggingface-hub==0.20.2', 'tokenizers==0.15.0', 'packaging==23.2', 'regex==2023.12.25', 'requests==2.31.0', 'bitsandbytes==0.42.0', 'accelerate==0.26.1'],
    license="Apache License 2.0",
    package_data={
        '':["LICENSE", "requirements.txt"]
    }
)