
from setuptools import setup, find_packages

setup(
    name="MofNeuroSim",
    version="0.1.0",
    description="A foundational framework for Spiking Neural Networks logic synthesis",
    author="MofNeuroSim Project",
    packages=find_packages(),  # This will find 'atomic_ops' and 'models'
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
