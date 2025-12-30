"""Setup script for hallucination detection framework."""
from setuptools import setup, find_packages

setup(
    name="hallucination-detection",
    version="0.1.0",
    description="Extensible hallucination detection framework based on lapeigvals",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],
        "api": ["dashscope>=1.14.0", "openai>=1.0.0"],
        "pipeline": ["dvc>=3.0.0"],
        "dev": ["pytest", "black", "flake8"],
    },
)
