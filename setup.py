"""
AI-Powered Lie Detection System — Package Setup
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="lie-detection-system",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@liedetect.ai",
    description="AI-Powered Multimodal Lie Detection System using Text & Voice Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lie-detection-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "liedetect-api=api.main:run",
            "liedetect-train=src.nlp.trainer:main",
            "liedetect-evaluate=src.evaluation.metrics:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="lie-detection nlp audio machine-learning deception multimodal",
)
