#!/usr/bin/env python3.12

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="novacron-sdk",
    version="1.0.0",
    author="NovaCron Team",
    author_email="team@novacron.io",
    description="Python SDK for NovaCron VM management platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khryptorgraphics/novacron",
    project_urls={
        "Bug Tracker": "https://github.com/khryptorgraphics/novacron/issues",
        "Documentation": "https://docs.novacron.io",
        "Source Code": "https://github.com/khryptorgraphics/novacron",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "novacron=novacron.cli:main",
        ],
    },
    keywords="vm management virtualization cloud computing infrastructure",
    include_package_data=True,
    zip_safe=False,
)