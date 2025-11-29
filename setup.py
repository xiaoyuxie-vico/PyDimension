"""
Setup script for PyDimension 2.0 package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "sympy>=1.9.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.12.0",
        "torch>=1.9.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.28.0",
    ]

# Find all config files
config_files = []
config_dir = Path(__file__).parent / "pydimension" / "configs"
if config_dir.exists():
    for config_file in config_dir.glob("*.json"):
        config_files.append(f"configs/{config_file.name}")

setup(
    name="pydimension",
    version="2.0.0",
    description="A comprehensive Python package for discovering dimensionless relationships in physical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PyDimension Team",
    author_email="pydimension@example.com",
    url="https://github.com/xiaoyuxie-vico/PyDimension",
    packages=find_packages(exclude=["tests", "tests.*", "*.tests", "*.tests.*"]),
    package_data={
        "pydimension": [
            "configs/*.json",
            "configs/*.md",
        ],
    },
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "pydimension-generate=pydimension.data_generation.__main__:main",
            "pydimension-preprocess=pydimension.data_preprocessing.__main__:main",
            "pydimension-analyze=pydimension.dimensional_analysis.__main__:main",
            "pydimension-filter=pydimension.constraint_filtering.__main__:main",
            "pydimension-optimize=pydimension.optimization_discovery.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="dimensional analysis, dimensionless groups, machine learning, physics, scaling laws",
    project_urls={
        "Bug Reports": "https://github.com/xiaoyuxie-vico/PyDimension/issues",
        "Source": "https://github.com/xiaoyuxie-vico/PyDimension",
        "Documentation": "https://github.com/xiaoyuxie-vico/PyDimension#readme",
    },
)

