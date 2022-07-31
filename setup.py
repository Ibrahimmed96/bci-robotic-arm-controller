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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bci-robotic-arm-controller",
    version="1.0.0",
    authors=["Ibrahim Mediouni", "Selim Ouirari"],
    author_email="ibrahim.mediouni@example.com",
    description="Control robotic arms using EEG brain signals via OpenBCI and ROS2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahimmediouni/bci-robotic-arm-controller",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "bci-stream=openbci_reader.stream_data:main",
            "bci-classify=signal_processing.eeg_classifier:main",
            "bci-control=ros2_publisher.arm_control_node:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    include_package_data=True,
)
# Updated on 2022-07-31