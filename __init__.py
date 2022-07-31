"""
BCI Robotic Arm Controller Package.

A comprehensive brain-computer interface system that enables real-time control
of robotic arms using EEG signals. This package integrates OpenBCI hardware,
advanced signal processing algorithms, and ROS2 communication protocols to
create a seamless brain-to-robot control pipeline.

The system is designed for research applications, assistive robotics, and
advanced human-machine interfaces where direct neural control is required.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

__version__ = "1.0.0"
__authors__ = ["Ibrahim Mediouni", "Selim Ouirari"]
__email__ = "ibrahim.mediouni@example.com"
__license__ = "MIT"

# Make key components available at package level
from . import openbci_reader
from . import signal_processing
from . import ros2_publisher

__all__ = [
    "openbci_reader",
    "signal_processing", 
    "ros2_publisher",
]