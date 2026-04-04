from setuptools import setup, find_packages

setup(
    name="shannon-b1",
    version="1.0.0",
    description="Shannon-b1: A lightweight GPT-style language model",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)