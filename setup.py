"""
Setup script for nnef-dists package.

This package provides neural network models for learning the log normalizer
of exponential family distributions, compatible with Hugging Face standards.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Neural Networks for Exponential Family Log Normalizers"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [
        "jax>=0.4.0",
        "flax>=0.7.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "blackjax>=0.9.0",
        "numpyro>=0.12.0",
    ]

setup(
    name="nnef-dists",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Networks for Exponential Family Log Normalizers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/nnef-dists",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "hf": [
            "transformers>=4.30.0",
            "huggingface-hub>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nnef-train=scripts.training.train:main",
            "nnef-eval=scripts.evaluation.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "machine-learning",
        "neural-networks", 
        "exponential-families",
        "log-normalizer",
        "jax",
        "flax",
        "hugging-face",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/nnef-dists/issues",
        "Source": "https://github.com/your-username/nnef-dists",
        "Documentation": "https://nnef-dists.readthedocs.io/",
    },
)
