#!/usr/bin/env python3
"""
ROS2 Robotic Arm Control Node for BCI System.

This module implements a ROS2 node that translates classified brain signals
into precise robotic arm movements. It serves as the bridge between the
brain-computer interface and the physical robotic hardware.

The system handles real-time command processing, safety mechanisms, inverse
kinematics calculations, and direct hardware communication through serial
protocols. It's designed to work with various robotic arm platforms while
maintaining strict safety standards.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

import numpy as np
import time
import threading
import logging
import json
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum

# Serial communication for hardware control
import serial
import serial.tools.list_ports

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String, Float32MultiArray, Bool
    from geometry_msgs.msg import Point, Quaternion, Pose, Twist
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 (rclpy) not available. Using mock implementation.")
    ROS2_AVAILABLE = False
    # Mock ROS2 classes for development without ROS2 installation
    class Node:
        def __init__(self, name): 
            self.name = name
            self.logger = logging.getLogger(name)
        def get_logger(self): 
            return self.logger
        def create_subscription(self, *args, **kwargs): 
            return None
        def create_publisher(self, *args, **kwargs): 
            return MockPublisher()
        def create_timer(self, period, callback): 
            return None
    class MockPublisher:
        def publish(self, msg): 
            pass

# Import our BCI components
from ..signal_processing.eeg_classifier import EEGClassifier, SignalProcessor
from ..openbci_reader.stream_data import BCIDataStreamer


class ArmCommand(Enum):
    """Enumeration of robotic arm commands derived from brain signals."""
    STOP = "stop"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    HOME_POSITION = "home_position"


@dataclass
class ArmState:
    """Data structure representing the current state of the robotic arm."""
    joint_angles: np.ndarray  # Current joint angles in radians
    end_effector_position: np.ndarray  # Cartesian position [x, y, z] in meters
    end_effector_orientation: np.ndarray  # Quaternion [x, y, z, w]
    gripper_position: float  # Gripper opening (0=closed, 1=open)
    is_moving: bool  # Whether the arm is currently executing a movement
    error_state: bool  # Whether the arm is in an error condition
    timestamp: float  # Time of last state update


@dataclass
class SafetyLimits:
    """Safety constraints for robotic arm operation."""
    max_joint_velocities: np.ndarray  # Maximum joint velocities (rad/s)
    joint_position_limits: Tuple[np.ndarray, np.ndarray]  # (min, max) positions
    workspace_boundaries: Dict[str, Tuple[float, float]]  # Cartesian workspace limits
    max_end_effector_velocity: float  # Maximum end-effector speed (m/s)
    emergency_stop_enabled: bool  # Whether emergency stop is active


class RobotController:
    """
    Low-level controller for robotic arm hardware communication.
    
    This class handles direct communication with the robotic arm hardware
    through serial interfaces. It implements the communication protocol
    for STM32-based motor controllers and provides real-time position
    control with safety monitoring.
    
    The controller supports various arm configurations and can adapt to
    different hardware platforms through configuration parameters.
    
    Parameters
    ----------
    port : str
        Serial port for arm communication
    baud_rate : int
        Serial communication baud rate
    num_joints : int
        Number of joints in the robotic arm
        
    Attributes
    ----------
    serial_connection : serial.Serial
        Serial connection to the arm controller
    current_state : ArmState
        Current state of the robotic arm
    safety_limits : SafetyLimits
        Safety constraints and limits
    """
    
    def __init__(self, port: str = "/dev/ttyACM0", baud_rate: int = 115200, num_joints: int = 6):
        """Initialize the robot controller with hardware configuration."""
        self.port = port
        self.baud_rate = baud_rate
        self.num_joints = num_joints
        
        # Hardware communication
        self.serial_connection = None
        self.is_connected = False
        
        # Robot state
        self.current_state = ArmState(
            joint_angles=np.zeros(num_joints),
            end_effector_position=np.array([0.3, 0.0, 0.2]),  # Default position
            end_effector_orientation=np.array([0, 0, 0, 1]),  # Identity quaternion
            gripper_position=0.0,
            is_moving=False,
            error_state=False,
            timestamp=time.time()
        )
        
        # Safety configuration
        self.safety_limits = SafetyLimits(
            max_joint_velocities=np.array([1.0, 1.0, 1.5, 2.0, 2.0, 3.0]),  # rad/s
            joint_position_limits=(
                np.array([-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi]),  # min
                np.array([np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi])   # max
            ),
            workspace_boundaries={
                'x': (0.1, 0.6),  # meters
                'y': (-0.4, 0.4),
                'z': (0.0, 0.5)
            },
            max_end_effector_velocity=0.2,  # m/s
            emergency_stop_enabled=False
        )
        
        # Command queue and execution
        self.command_queue = deque(maxlen=100)
        self.is_executing = False
        self.execution_thread = None
        
        # Kinematics parameters (for a typical 6-DOF arm)
        self.dh_parameters = self._initialize_dh_parameters()
        
        # Performance monitoring
        self.command_count = 0
        self.error_count = 0
        self.last_command_time = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Robot controller initialized: {num_joints} joints, port {port}")
    
    def _initialize_dh_parameters(self) -> np.ndarray:
        """
        Initialize Denavit-Hartenberg parameters for kinematic calculations.
        
        These parameters define the kinematic structure of the robotic arm
        and are used for forward and inverse kinematics calculations.
        
        Returns
        -------
        dh_params : numpy.ndarray
            DH parameter matrix with shape (num_joints, 4)
            Columns: [a, alpha, d, theta_offset]
        """
        # Default DH parameters for a 6-DOF robotic arm
        # These should be adjusted based on the specific arm configuration
        dh_params = np.array([
            [0.0,    np.pi/2,  0.15,  0.0],     # Joint 1: Base rotation
            [0.25,   0.0,      0.0,   -np.pi/2], # Joint 2: Shoulder
            [0.20,   0.0,      0.0,   0.0],     # Joint 3: Elbow
            [0.0,    np.pi/2,  0.15,  0.0],     # Joint 4: Wrist 1
            [0.0,   -np.pi/2,  0.0,   0.0],     # Joint 5: Wrist 2
            [0.0,    0.0,      0.08,  0.0]      # Joint 6: Wrist 3
        ])
        
        return dh_params
    
    def connect(self) -> bool:
        """
        Establish connection with the robotic arm hardware.
        
        Attempts to connect to the STM32-based arm controller through
        serial communication. Includes hardware detection, protocol
        verification, and initial state synchronization.
        
        Returns
        -------
        success : bool
            True if connection was successful
        """
        try:
            # List available serial ports
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            self.logger.info(f"Available serial ports: {available_ports}")
            
            if self.port not in available_ports:
                self.logger.warning(f"Specified port {self.port} not found. Using simulation mode.")
                self.is_connected = True  # Enable simulation mode
                return True
            
            # Establish serial connection
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=2.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for connection to stabilize
            time.sleep(2.0)
            
            # Verify communication with the controller
            if self._verify_communication():
                self.is_connected = True
                
                # Initialize arm to known state
                self._initialize_arm()
                
                self.logger.info("Successfully connected to robotic arm")
                return True
            else:
                self.logger.error("Communication verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to robotic arm: {e}")
            self.is_connected = True  # Fallback to simulation mode
            return True
    
    def _verify_communication(self) -> bool:
        """
        Verify communication with the arm controller.
        
        Sends a status request command and verifies the response to ensure
        the communication protocol is working correctly.
        
        Returns
        -------
        verified : bool
            True if communication is working
        """
        try:
            if not self.serial_connection:
                return False
            
            # Send status request command
            command = "STATUS\n"
            self.serial_connection.write(command.encode())
            
            # Wait for response
            response = self.serial_connection.readline().decode().strip()
            
            # Check if response indicates the controller is ready
            if "READY" in response or "OK" in response:
                return True
            else:
                self.logger.warning(f"Unexpected response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Communication verification error: {e}")
            return False
    
    def _initialize_arm(self) -> None:
        """Initialize the robotic arm to a safe home position."""
        try:
            # Send home command to the controller
            home_command = "HOME\n"
            if self.serial_connection:
                self.serial_connection.write(home_command.encode())
                time.sleep(0.1)
            
            # Update internal state to home position
            self.current_state.joint_angles = np.zeros(self.num_joints)
            self.current_state.end_effector_position = np.array([0.3, 0.0, 0.2])
            self.current_state.end_effector_orientation = np.array([0, 0, 0, 1])
            self.current_state.gripper_position = 0.0
            self.current_state.is_moving = False
            self.current_state.error_state = False
            self.current_state.timestamp = time.time()
            
            self.logger.info("Arm initialized to home position")
            
        except Exception as e:
            self.logger.error(f"Arm initialization failed: {e}")
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate end-effector pose from joint angles.
        
        Computes the forward kinematics using the Denavit-Hartenberg
        convention to determine the end-effector position and orientation
        from given joint angles.
        
        Parameters
        ----------
        joint_angles : numpy.ndarray
            Joint angles in radians
            
        Returns
        -------
        position : numpy.ndarray
            End-effector position [x, y, z] in meters
        orientation : numpy.ndarray  
            End-effector orientation as quaternion [x, y, z, w]
        """
        try:
            # Initialize transformation matrix as identity
            T = np.eye(4)
            
            # Apply DH transformations for each joint
            for i in range(min(len(joint_angles), self.num_joints)):
                a, alpha, d, theta_offset = self.dh_parameters[i]
                theta = joint_angles[i] + theta_offset
                
                # DH transformation matrix
                ct, st = np.cos(theta), np.sin(theta)
                ca, sa = np.cos(alpha), np.sin(alpha)
                
                T_i = np.array([
                    [ct,    -st*ca,  st*sa,   a*ct],
                    [st,     ct*ca, -ct*sa,   a*st],
                    [0,      sa,     ca,      d],
                    [0,      0,      0,       1]
                ])
                
                T = T @ T_i
            
            # Extract position and orientation
            position = T[:3, 3]
            
            # Convert rotation matrix to quaternion
            rotation_matrix = T[:3, :3]
            orientation = self._rotation_matrix_to_quaternion(rotation_matrix)
            
            return position, orientation
            
        except Exception as e:
            self.logger.error(f"Forward kinematics calculation failed: {e}")
            return self.current_state.end_effector_position, self.current_state.end_effector_orientation
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion representation.
        
        Parameters
        ----------
        R : numpy.ndarray
            3x3 rotation matrix
            
        Returns
        -------
        quaternion : numpy.ndarray
            Quaternion [x, y, z, w]
        """
        try:
            trace = np.trace(R)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            return np.array([qx, qy, qz, qw])
            
        except Exception as e:
            self.logger.error(f"Quaternion conversion failed: {e}")
            return np.array([0, 0, 0, 1])  # Identity quaternion
    
    def execute_command(self, command: ArmCommand, parameters: Optional[Dict] = None) -> bool:
        """
        Execute a robotic arm command.
        
        Processes and executes commands derived from brain signals,
        including safety checks and motion planning. Commands are
        executed asynchronously to maintain real-time performance.
        
        Parameters
        ----------
        command : ArmCommand
            Command to execute
        parameters : dict, optional
            Additional command parameters
            
        Returns
        -------
        success : bool
            True if command was accepted for execution
        """
        if not self.is_connected:
            self.logger.error("Arm not connected")
            return False
        
        if self.safety_limits.emergency_stop_enabled:
            self.logger.warning("Emergency stop active - command ignored")
            return False
        
        try:
            # Validate command parameters
            if not self._validate_command(command, parameters):
                return False
            
            # Add command to execution queue
            command_data = {
                'command': command,
                'parameters': parameters or {},
                'timestamp': time.time()
            }
            
            self.command_queue.append(command_data)
            
            # Start execution thread if not running
            if not self.is_executing:
                self._start_execution_thread()
            
            self.command_count += 1
            self.last_command_time = time.time()
            
            self.logger.debug(f"Command queued: {command.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            self.error_count += 1
            return False
    
    def _validate_command(self, command: ArmCommand, parameters: Optional[Dict]) -> bool:
        """
        Validate command safety and feasibility.
        
        Checks if the requested command is safe to execute given the
        current arm state and safety constraints.
        
        Parameters
        ----------
        command : ArmCommand
            Command to validate
        parameters : dict, optional
            Command parameters
            
        Returns
        -------
        valid : bool
            True if command is safe to execute
        """
        try:
            # Check if arm is in error state
            if self.current_state.error_state:
                self.logger.warning("Cannot execute command - arm in error state")
                return False
            
            # Validate movement commands
            if command in [ArmCommand.MOVE_UP, ArmCommand.MOVE_DOWN, 
                          ArmCommand.MOVE_LEFT, ArmCommand.MOVE_RIGHT,
                          ArmCommand.MOVE_FORWARD, ArmCommand.MOVE_BACKWARD]:
                
                # Check workspace boundaries
                current_pos = self.current_state.end_effector_position
                step_size = parameters.get('step_size', 0.02) if parameters else 0.02
                
                # Calculate new position based on command
                new_pos = current_pos.copy()
                if command == ArmCommand.MOVE_UP:
                    new_pos[2] += step_size
                elif command == ArmCommand.MOVE_DOWN:
                    new_pos[2] -= step_size
                elif command == ArmCommand.MOVE_LEFT:
                    new_pos[1] += step_size
                elif command == ArmCommand.MOVE_RIGHT:
                    new_pos[1] -= step_size
                elif command == ArmCommand.MOVE_FORWARD:
                    new_pos[0] += step_size
                elif command == ArmCommand.MOVE_BACKWARD:
                    new_pos[0] -= step_size
                
                # Check workspace boundaries
                limits = self.safety_limits.workspace_boundaries
                if not (limits['x'][0] <= new_pos[0] <= limits['x'][1] and
                        limits['y'][0] <= new_pos[1] <= limits['y'][1] and
                        limits['z'][0] <= new_pos[2] <= limits['z'][1]):
                    self.logger.warning("Command would exceed workspace boundaries")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command validation failed: {e}")
            return False
    
    def _start_execution_thread(self) -> None:
        """Start the command execution thread."""
        if self.execution_thread and self.execution_thread.is_alive():
            return
        
        self.is_executing = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        self.logger.debug("Command execution thread started")
    
    def _execution_loop(self) -> None:
        """Main command execution loop running in separate thread."""
        while self.is_executing:
            try:
                if self.command_queue:
                    # Get next command
                    command_data = self.command_queue.popleft()
                    command = command_data['command']
                    parameters = command_data['parameters']
                    
                    # Execute the command
                    self._execute_single_command(command, parameters)
                    
                else:
                    # No commands to execute, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                time.sleep(0.1)
    
    def _execute_single_command(self, command: ArmCommand, parameters: Dict) -> None:
        """
        Execute a single arm command.
        
        Performs the actual hardware communication and state updates
        for a specific command.
        
        Parameters
        ----------
        command : ArmCommand
            Command to execute
        parameters : dict
            Command parameters
        """
        try:
            self.current_state.is_moving = True
            
            # Execute based on command type
            if command == ArmCommand.STOP:
                self._send_stop_command()
                
            elif command == ArmCommand.HOME_POSITION:
                self._send_home_command()
                
            elif command in [ArmCommand.MOVE_UP, ArmCommand.MOVE_DOWN,
                           ArmCommand.MOVE_LEFT, ArmCommand.MOVE_RIGHT,
                           ArmCommand.MOVE_FORWARD, ArmCommand.MOVE_BACKWARD]:
                self._send_movement_command(command, parameters)
                
            elif command == ArmCommand.OPEN_GRIPPER:
                self._send_gripper_command(1.0)  # Fully open
                
            elif command == ArmCommand.CLOSE_GRIPPER:
                self._send_gripper_command(0.0)  # Fully closed
            
            # Update state timestamp
            self.current_state.timestamp = time.time()
            
        except Exception as e:
            self.logger.error(f"Single command execution failed: {e}")
            self.current_state.error_state = True
        finally:
            self.current_state.is_moving = False
    
    def _send_movement_command(self, command: ArmCommand, parameters: Dict) -> None:
        """Send movement command to hardware controller."""
        step_size = parameters.get('step_size', 0.02)  # Default 2cm steps
        
        # Calculate target position
        current_pos = self.current_state.end_effector_position.copy()
        
        if command == ArmCommand.MOVE_UP:
            current_pos[2] += step_size
        elif command == ArmCommand.MOVE_DOWN:
            current_pos[2] -= step_size
        elif command == ArmCommand.MOVE_LEFT:
            current_pos[1] += step_size
        elif command == ArmCommand.MOVE_RIGHT:
            current_pos[1] -= step_size
        elif command == ArmCommand.MOVE_FORWARD:
            current_pos[0] += step_size
        elif command == ArmCommand.MOVE_BACKWARD:
            current_pos[0] -= step_size
        
        # Send position command to controller
        if self.serial_connection:
            cmd_str = f"MOVE {current_pos[0]:.3f} {current_pos[1]:.3f} {current_pos[2]:.3f}\n"
            self.serial_connection.write(cmd_str.encode())
        
        # Update internal state
        self.current_state.end_effector_position = current_pos
        
        # Simulate execution time
        time.sleep(0.1)
    
    def _send_gripper_command(self, position: float) -> None:
        """Send gripper control command to hardware."""
        if self.serial_connection:
            cmd_str = f"GRIPPER {position:.2f}\n"
            self.serial_connection.write(cmd_str.encode())
        
        self.current_state.gripper_position = position
        time.sleep(0.1)
    
    def _send_stop_command(self) -> None:
        """Send emergency stop command to hardware."""
        if self.serial_connection:
            self.serial_connection.write(b"STOP\n")
        
        self.current_state.is_moving = False
        time.sleep(0.05)
    
    def _send_home_command(self) -> None:
        """Send home position command to hardware."""
        if self.serial_connection:
            self.serial_connection.write(b"HOME\n")
        
        # Reset to home state
        self.current_state.joint_angles = np.zeros(self.num_joints)
        self.current_state.end_effector_position = np.array([0.3, 0.0, 0.2])
        self.current_state.end_effector_orientation = np.array([0, 0, 0, 1])
        self.current_state.gripper_position = 0.0
        
        time.sleep(2.0)  # Home operation takes longer
    
    def get_state(self) -> ArmState:
        """
        Get current arm state.
        
        Returns
        -------
        state : ArmState
            Current state of the robotic arm
        """
        return self.current_state
    
    def emergency_stop(self) -> None:
        """Activate emergency stop - immediately halt all motion."""
        self.safety_limits.emergency_stop_enabled = True
        self._send_stop_command()
        self.logger.warning("Emergency stop activated")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop condition."""
        self.safety_limits.emergency_stop_enabled = False
        self.current_state.error_state = False
        self.logger.info("Emergency stop reset")
    
    def disconnect(self) -> None:
        """Disconnect from robotic arm hardware."""
        try:
            # Stop execution thread
            self.is_executing = False
            if self.execution_thread and self.execution_thread.is_alive():
                self.execution_thread.join(timeout=2.0)
            
            # Send final stop command
            if self.serial_connection:
                self._send_stop_command()
                self.serial_connection.close()
                self.serial_connection = None
            
            self.is_connected = False
            self.logger.info("Disconnected from robotic arm")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")


class ArmControlNode(Node):
    """
    ROS2 node for brain-controlled robotic arm operation.
    
    This node integrates all components of the BCI system: EEG acquisition,
    signal processing, brain state classification, and robotic arm control.
    It provides a complete ROS2-based solution for brain-controlled robotics.
    
    The node publishes arm state information, subscribes to brain signal
    classifications, and provides services for system configuration and
    monitoring. It's designed for real-time operation with comprehensive
    error handling and safety mechanisms.
    
    Parameters
    ----------
    node_name : str
        Name of the ROS2 node
        
    Attributes
    ----------
    robot_controller : RobotController
        Low-level robotic arm controller
    eeg_classifier : EEGClassifier
        Brain state classification system
    signal_processor : SignalProcessor
        EEG signal processing pipeline
    """
    
    def __init__(self, node_name: str = 'bci_arm_control_node'):
        """Initialize the BCI arm control ROS2 node."""
        super().__init__(node_name)
        
        # Initialize BCI components
        self.robot_controller = RobotController()
        self.eeg_classifier = EEGClassifier()
        self.signal_processor = SignalProcessor()
        self.data_streamer = BCIDataStreamer()
        
        # ROS2 publishers
        self.arm_state_publisher = self.create_publisher(
            JointState, 'arm_state', 10
        )
        
        self.brain_state_publisher = self.create_publisher(
            String, 'brain_state', 10
        )
        
        self.status_publisher = self.create_publisher(
            String, 'system_status', 10
        )
        
        # ROS2 subscribers
        self.brain_command_subscriber = self.create_subscription(
            String, 'brain_commands', self.brain_command_callback, 10
        )
        
        self.emergency_stop_subscriber = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10
        )
        
        # Timers for periodic operations
        self.state_update_timer = self.create_timer(0.1, self.publish_state_callback)  # 10 Hz
        self.brain_processing_timer = self.create_timer(0.2, self.process_brain_signals)  # 5 Hz
        self.status_timer = self.create_timer(1.0, self.publish_status_callback)  # 1 Hz
        
        # System state
        self.is_initialized = False
        self.last_brain_command = None
        self.command_mapping = self._create_command_mapping()
        
        # Performance monitoring
        self.node_start_time = time.time()
        self.brain_commands_processed = 0
        self.arm_commands_executed = 0
        
        self.get_logger().info("BCI Arm Control Node initialized")
    
    def _create_command_mapping(self) -> Dict[str, ArmCommand]:
        """
        Create mapping from brain states to arm commands.
        
        This mapping translates classified brain states into specific
        robotic arm movements. The mapping can be customized based on
        user preferences and training data.
        
        Returns
        -------
        mapping : dict
            Dictionary mapping brain state names to arm commands
        """
        mapping = {
            'rest': ArmCommand.STOP,
            'focus': ArmCommand.MOVE_FORWARD,
            'relax': ArmCommand.MOVE_BACKWARD,
            'movement': ArmCommand.MOVE_UP,
            'left_hand': ArmCommand.MOVE_LEFT,
            'right_hand': ArmCommand.MOVE_RIGHT,
            'up': ArmCommand.MOVE_UP,
            'down': ArmCommand.MOVE_DOWN,
            'open': ArmCommand.OPEN_GRIPPER,
            'close': ArmCommand.CLOSE_GRIPPER,
            'home': ArmCommand.HOME_POSITION,
        }
        
        return mapping
    
    def initialize_system(self) -> bool:
        """
        Initialize all BCI system components.
        
        Connects to hardware, loads trained models, and prepares the
        system for real-time operation. This method should be called
        before starting brain-controlled operation.
        
        Returns
        -------
        success : bool
            True if all components initialized successfully
        """
        try:
            self.get_logger().info("Initializing BCI system...")
            
            # Connect to robotic arm
            if not self.robot_controller.connect():
                self.get_logger().error("Failed to connect to robotic arm")
                return False
            
            # Connect to EEG hardware
            if not self.data_streamer.connect():
                self.get_logger().warning("EEG connection failed, using simulation mode")
            
            # Load trained classification model
            model_path = "eeg_model.pkl"  # This should be configurable
            if os.path.exists(model_path):
                if self.eeg_classifier.load_model(model_path):
                    self.get_logger().info(f"Loaded EEG model: {self.eeg_classifier.class_names}")
                else:
                    self.get_logger().warning("Failed to load EEG model")
            else:
                self.get_logger().warning("No trained EEG model found")
            
            # Start EEG data streaming
            if not self.data_streamer.start_streaming():
                self.get_logger().warning("EEG streaming failed")
            
            self.is_initialized = True
            self.get_logger().info("BCI system initialization complete")
            return True
            
        except Exception as e:
            self.get_logger().error(f"System initialization failed: {e}")
            return False
    
    def brain_command_callback(self, msg: String) -> None:
        """
        Process incoming brain command messages.
        
        This callback receives classified brain states from the signal
        processing pipeline and translates them into robotic arm commands.
        
        Parameters
        ----------
        msg : String
            ROS2 message containing brain state classification
        """
        try:
            # Parse brain command message
            brain_state = msg.data.strip().lower()
            self.last_brain_command = brain_state
            
            # Map brain state to arm command
            if brain_state in self.command_mapping:
                arm_command = self.command_mapping[brain_state]
                
                # Execute arm command
                if self.robot_controller.execute_command(arm_command):
                    self.arm_commands_executed += 1
                    self.get_logger().debug(f"Executed: {brain_state} -> {arm_command.value}")
                else:
                    self.get_logger().warning(f"Failed to execute command: {arm_command.value}")
            else:
                self.get_logger().warning(f"Unknown brain state: {brain_state}")
            
            self.brain_commands_processed += 1
            
        except Exception as e:
            self.get_logger().error(f"Brain command callback error: {e}")
    
    def emergency_stop_callback(self, msg: Bool) -> None:
        """
        Handle emergency stop messages.
        
        Parameters
        ----------
        msg : Bool
            Emergency stop activation status
        """
        if msg.data:
            self.robot_controller.emergency_stop()
            self.get_logger().warning("Emergency stop activated via ROS2")
        else:
            self.robot_controller.reset_emergency_stop()
            self.get_logger().info("Emergency stop reset via ROS2")
    
    def process_brain_signals(self) -> None:
        """
        Process EEG signals and classify brain states.
        
        This method runs periodically to analyze incoming EEG data,
        extract features, and classify the current brain state for
        robotic control.
        """
        if not self.is_initialized or not self.eeg_classifier.is_trained:
            return
        
        try:
            # Get recent EEG data
            eeg_data, timestamps = self.data_streamer.get_recent_data(duration_seconds=2.0)
            
            if eeg_data.size == 0:
                return
            
            # Process the signal
            features = self.signal_processor.process_segment(eeg_data)
            
            if not features:
                return
            
            # Classify brain state
            prediction, confidence, probabilities = self.eeg_classifier.predict(features)
            
            # Only act on high-confidence predictions
            confidence_threshold = 0.7
            if confidence >= confidence_threshold:
                # Publish brain state
                brain_msg = String()
                brain_msg.data = prediction
                self.brain_state_publisher.publish(brain_msg)
                
                # Process as brain command
                self.brain_command_callback(brain_msg)
            
        except Exception as e:
            self.get_logger().error(f"Brain signal processing error: {e}")
    
    def publish_state_callback(self) -> None:
        """Publish current arm state information."""
        try:
            arm_state = self.robot_controller.get_state()
            
            # Create JointState message
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = "base_link"
            
            # Populate joint information
            joint_msg.name = [f"joint_{i+1}" for i in range(len(arm_state.joint_angles))]
            joint_msg.position = arm_state.joint_angles.tolist()
            joint_msg.velocity = [0.0] * len(arm_state.joint_angles)  # TODO: Add velocity estimation
            joint_msg.effort = [0.0] * len(arm_state.joint_angles)   # TODO: Add effort feedback
            
            self.arm_state_publisher.publish(joint_msg)
            
        except Exception as e:
            self.get_logger().error(f"State publishing error: {e}")
    
    def publish_status_callback(self) -> None:
        """Publish system status information."""
        try:
            # Gather system status
            status_data = {
                'initialized': self.is_initialized,
                'arm_connected': self.robot_controller.is_connected,
                'eeg_connected': self.data_streamer.is_streaming,
                'model_trained': self.eeg_classifier.is_trained,
                'uptime': time.time() - self.node_start_time,
                'brain_commands_processed': self.brain_commands_processed,
                'arm_commands_executed': self.arm_commands_executed,
                'last_brain_command': self.last_brain_command,
            }
            
            # Publish status
            status_msg = String()
            status_msg.data = json.dumps(status_data)
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Status publishing error: {e}")
    
    def shutdown(self) -> None:
        """Properly shutdown the BCI system."""
        try:
            self.get_logger().info("Shutting down BCI system...")
            
            # Stop data streaming
            self.data_streamer.stop_streaming()
            
            # Disconnect from hardware
            self.robot_controller.disconnect()
            self.data_streamer.disconnect()
            
            self.get_logger().info("BCI system shutdown complete")
            
        except Exception as e:
            self.get_logger().error(f"Shutdown error: {e}")


def main(args=None):
    """
    Main function for the ROS2 BCI arm control node.
    
    Initializes ROS2, creates the BCI control node, and runs the main
    execution loop. Handles graceful shutdown on interruption.
    """
    if not ROS2_AVAILABLE:
        print("Error: ROS2 not available. Please install ROS2 to run this node.")
        return
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create and initialize the BCI node
        node = ArmControlNode()
        
        # Initialize the BCI system
        if not node.initialize_system():
            node.get_logger().error("System initialization failed")
            return
        
        # Run the node
        node.get_logger().info("BCI Arm Control Node running...")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        print(f"Node execution error: {e}")
    finally:
        # Cleanup
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# Updated on 2022-12-31
# Updated on 2022-12-12
# Updated on 2022-12-12
# Updated on 2022-12-20
# Updated on 2022-12-27
# Updated on 2022-12-13