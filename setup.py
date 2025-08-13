#!/usr/bin/env python3
"""
Setup script for BCI Robotic Arm Controller.

This package provides a complete brain-computer interface system for controlling
robotic arms using EEG signals from OpenBCI devices. The system processes
real-time brainwave data, classifies mental states, and translates them into
precise robotic movements.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Extract dependencies from requirements.txt, filtering out comments and empty lines
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    # Basic package metadata
    name="bci-robotic-arm-controller",
    version="2.0.0",  # Semantic versioning: MAJOR.MINOR.PATCH
    authors=["Ibrahim Mediouni", "Selim Ouirari"],
    author_email="ibrahim.mediouni@example.com",
    
    # Package description and documentation
    description="Control robotic arms using EEG brain signals via OpenBCI and ROS2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahimmediouni/bci-robotic-arm-controller",
    
    # Package discovery - automatically find all Python packages
    packages=find_packages(),
    
    # PyPI classifiers for package categorization and discoverability
    classifiers=[
        "Development Status :: 4 - Beta",  # Stable for production use with ongoing development
        "Intended Audience :: Science/Research",  # Target audience: researchers and academics
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",  # BCI/HMI domain
        "Topic :: Scientific/Engineering :: Medical Science Apps.",  # Assistive technology
        "License :: OSI Approved :: MIT License",  # Open source MIT license
        "Programming Language :: Python :: 3",  # Python 3 support
        "Programming Language :: Python :: 3.8",  # Specific version compatibility
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",  # Cross-platform compatibility
    ],
    
    # Version compatibility and dependencies
    python_requires=">=3.8",  # Minimum Python version for modern features
    install_requires=requirements,  # Runtime dependencies from requirements.txt
    
    # Optional development dependencies for testing and documentation
    extras_require={
        "dev": [
            "pytest>=6.0",          # Testing framework with fixtures and assertions
            "pytest-cov>=2.0",      # Test coverage reporting
            "black>=21.0",          # Code formatting for consistent style
            "flake8>=3.9",          # Linting for code quality checks
            "sphinx>=4.0",          # Documentation generation from docstrings
            "sphinx-rtd-theme>=0.5", # Read the Docs theme for professional documentation
        ],
    },
    
    # Console script entry points for direct command-line usage
    entry_points={
        "console_scripts": [
            "bci-stream=openbci_reader.stream_data:main",      # EEG data streaming utility
            "bci-classify=signal_processing.eeg_classifier:main",  # Signal classification tool
            "bci-control=ros2_publisher.arm_control_node:main",    # ROS2 robotic arm controller
        ],
    },
    
    # Additional package data and resource files
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],  # Include configuration and data files
    },
    include_package_data=True,  # Include files specified in MANIFEST.in
)
