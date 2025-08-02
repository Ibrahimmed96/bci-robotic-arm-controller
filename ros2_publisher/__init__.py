"""
ROS2 Robotic Arm Control Module.

This module implements ROS2 nodes for translating classified brain signals
into precise robotic arm movements. It handles the communication between
the BCI system and robotic hardware through standardized ROS2 interfaces.

The module includes safety mechanisms, inverse kinematics calculations, and
direct hardware communication protocols for various robotic arm platforms.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

from .arm_control_node import ArmControlNode, RobotController

__all__ = [
    "ArmControlNode", 
    "RobotController",
]
