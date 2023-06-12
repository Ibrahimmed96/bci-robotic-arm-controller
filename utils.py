#!/usr/bin/env python3
"""
Utility functions and helper classes for the BCI Robotic Arm Controller.

This module provides common utilities, data structures, and helper functions
used throughout the BCI system. It includes configuration management,
logging setup, data validation, and mathematical utilities.

These utilities support the main BCI components with reusable functionality
that maintains consistency across the entire system.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

import numpy as np
import json
import yaml
import logging
import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import configparser


@dataclass
class BCIConfig:
    """
    Configuration data structure for the BCI system.
    
    This class holds all configuration parameters for the BCI robotic arm
    controller, including hardware settings, signal processing parameters,
    and safety limits.
    
    Attributes
    ----------
    eeg_config : dict
        EEG acquisition and processing parameters
    arm_config : dict
        Robotic arm hardware and control settings
    classification_config : dict
        Machine learning and classification parameters
    safety_config : dict
        Safety limits and emergency procedures
    """
    
    # EEG Configuration
    eeg_sampling_rate: float = 250.0
    eeg_channels: int = 8
    eeg_port: str = "/dev/ttyUSB0"
    eeg_baud_rate: int = 115200
    eeg_daisy_mode: bool = False
    
    # Signal Processing Configuration
    filter_lowcut: float = 1.0
    filter_highcut: float = 45.0
    notch_frequency: float = 50.0
    notch_quality: float = 30.0
    
    # Classification Configuration
    model_type: str = "rf"
    feature_scaling: bool = True
    confidence_threshold: float = 0.7
    classification_window: float = 2.0
    
    # Robotic Arm Configuration
    arm_port: str = "/dev/ttyACM0"
    arm_baud_rate: int = 115200
    arm_joints: int = 6
    arm_step_size: float = 0.02
    
    # Safety Configuration
    max_joint_velocity: float = 1.0
    workspace_x_limits: Tuple[float, float] = (0.1, 0.6)
    workspace_y_limits: Tuple[float, float] = (-0.4, 0.4)
    workspace_z_limits: Tuple[float, float] = (0.0, 0.5)
    emergency_stop_enabled: bool = False
    
    # System Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    config_file: Optional[str] = None


class ConfigManager:
    """
    Configuration management system for the BCI application.
    
    This class handles loading, saving, and validating configuration
    parameters from various sources including YAML files, JSON files,
    and environment variables.
    
    The manager supports configuration inheritance and validation to
    ensure system reliability and ease of deployment.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file
        
    Attributes
    ----------
    config : BCIConfig
        Current system configuration
    config_file : str
        Path to the active configuration file
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager."""
        self.config = BCIConfig()
        self.config_file = config_path
        self.logger = logging.getLogger(__name__)
        
        # Default configuration search paths
        self.search_paths = [
            "./config.yaml",
            "./config.json", 
            "~/.bci_config.yaml",
            "/etc/bci/config.yaml"
        ]
        
        # Load configuration if path provided or found
        if config_path:
            self.load_config(config_path)
        else:
            self._auto_load_config()
    
    def _auto_load_config(self) -> None:
        """Automatically search for and load configuration files."""
        for path in self.search_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                try:
                    self.load_config(expanded_path)
                    self.logger.info(f"Loaded configuration from {expanded_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {expanded_path}: {e}")
        
        self.logger.info("No configuration file found, using defaults")
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Supports YAML and JSON configuration files. The configuration
        is merged with existing settings, allowing partial updates.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
            
        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        ValueError
            If the configuration file format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path}")
            
            # Update configuration with loaded data
            self._update_config(config_data)
            self.config_file = config_path
            
            # Validate the loaded configuration
            self._validate_config()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _update_config(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration with new data.
        
        Recursively updates the configuration object with values from
        the provided dictionary, preserving existing values for keys
        not present in the new data.
        
        Parameters
        ----------
        config_data : dict
            Configuration data to merge
        """
        for key, value in config_data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Checks that all configuration values are within acceptable ranges
        and that required dependencies are satisfied.
        
        Raises
        ------
        ValueError
            If configuration validation fails
        """
        # Validate EEG configuration
        if self.config.eeg_sampling_rate <= 0:
            raise ValueError("EEG sampling rate must be positive")
        
        if self.config.eeg_channels < 1 or self.config.eeg_channels > 32:
            raise ValueError("EEG channels must be between 1 and 32")
        
        # Validate signal processing configuration
        if self.config.filter_lowcut >= self.config.filter_highcut:
            raise ValueError("Filter low cutoff must be less than high cutoff")
        
        if self.config.notch_frequency <= 0:
            raise ValueError("Notch frequency must be positive")
        
        # Validate classification configuration
        if not 0 < self.config.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.config.classification_window <= 0:
            raise ValueError("Classification window must be positive")
        
        # Validate arm configuration
        if self.config.arm_joints < 3 or self.config.arm_joints > 10:
            raise ValueError("Arm joints must be between 3 and 10")
        
        if self.config.arm_step_size <= 0:
            raise ValueError("Arm step size must be positive")
        
        # Validate safety configuration
        if self.config.max_joint_velocity <= 0:
            raise ValueError("Maximum joint velocity must be positive")
        
        # Validate workspace limits
        x_min, x_max = self.config.workspace_x_limits
        y_min, y_max = self.config.workspace_y_limits
        z_min, z_max = self.config.workspace_z_limits
        
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Workspace limits must have min < max")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Parameters
        ----------
        config_path : str, optional
            Path to save the configuration. If None, uses the current config file.
        """
        save_path = config_path or self.config_file
        
        if not save_path:
            raise ValueError("No configuration file path specified")
        
        try:
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif save_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {save_path}")
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> BCIConfig:
        """
        Get current configuration.
        
        Returns
        -------
        config : BCIConfig
            Current system configuration
        """
        return self.config
    
    def update_parameter(self, parameter: str, value: Any) -> None:
        """
        Update a single configuration parameter.
        
        Parameters
        ----------
        parameter : str
            Name of the parameter to update
        value : Any
            New value for the parameter
        """
        if hasattr(self.config, parameter):
            old_value = getattr(self.config, parameter)
            setattr(self.config, parameter, value)
            
            try:
                self._validate_config()
                self.logger.info(f"Updated {parameter}: {old_value} -> {value}")
            except ValueError as e:
                # Revert change if validation fails
                setattr(self.config, parameter, old_value)
                raise ValueError(f"Invalid parameter value: {e}")
        else:
            raise ValueError(f"Unknown parameter: {parameter}")


class DataValidator:
    """
    Data validation utilities for EEG and robotic control data.
    
    This class provides methods to validate data integrity, detect
    anomalies, and ensure data quality throughout the BCI pipeline.
    
    Methods include statistical validation, range checking, and
    consistency verification for various data types used in the system.
    """
    
    @staticmethod
    def validate_eeg_data(data: np.ndarray, sampling_rate: float = 250.0) -> Dict[str, Any]:
        """
        Validate EEG data quality and integrity.
        
        Checks for common EEG data issues including saturation, noise,
        disconnected electrodes, and signal artifacts.
        
        Parameters
        ----------
        data : numpy.ndarray
            EEG data with shape (samples, channels)
        sampling_rate : float
            Sampling frequency in Hz
            
        Returns
        -------
        validation_result : dict
            Dictionary containing validation results and quality metrics
        """
        if data.size == 0:
            return {'valid': False, 'error': 'Empty data array'}
        
        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        samples, channels = data.shape
        
        result = {
            'valid': True,
            'warnings': [],
            'metrics': {},
            'channel_quality': []
        }
        
        try:
            # Check for saturation (clipped signals)
            saturation_threshold = 100.0  # microvolts
            saturated_samples = np.sum(np.abs(data) >= saturation_threshold, axis=0)
            saturation_rate = saturated_samples / samples
            
            result['metrics']['saturation_rate'] = saturation_rate.tolist()
            
            if np.any(saturation_rate > 0.1):  # More than 10% saturated
                result['warnings'].append('High saturation detected in some channels')
            
            # Check for flat lines (disconnected electrodes)
            variance_threshold = 0.1  # microvolts^2
            channel_variance = np.var(data, axis=0)
            flat_channels = channel_variance < variance_threshold
            
            result['metrics']['channel_variance'] = channel_variance.tolist()
            
            if np.any(flat_channels):
                flat_indices = np.where(flat_channels)[0].tolist()
                result['warnings'].append(f'Flat signal detected in channels: {flat_indices}')
            
            # Check for excessive noise
            noise_threshold = 50.0  # microvolts RMS
            rms_values = np.sqrt(np.mean(data**2, axis=0))
            noisy_channels = rms_values > noise_threshold
            
            result['metrics']['rms_values'] = rms_values.tolist()
            
            if np.any(noisy_channels):
                noisy_indices = np.where(noisy_channels)[0].tolist()
                result['warnings'].append(f'High noise detected in channels: {noisy_indices}')
            
            # Check data continuity (detect dropouts)
            if samples > 1:
                time_diffs = np.diff(np.arange(samples)) / sampling_rate
                expected_interval = 1.0 / sampling_rate
                dropout_threshold = expected_interval * 2  # Allow 2x normal interval
                
                dropouts = np.sum(time_diffs > dropout_threshold)
                result['metrics']['dropout_count'] = int(dropouts)
                
                if dropouts > 0:
                    result['warnings'].append(f'Data dropouts detected: {dropouts}')
            
            # Calculate overall quality score for each channel
            for ch in range(channels):
                quality_score = 1.0
                
                # Penalize saturation
                quality_score *= (1.0 - min(saturation_rate[ch], 1.0))
                
                # Penalize flat signals
                if flat_channels[ch]:
                    quality_score *= 0.1
                
                # Penalize excessive noise
                if noisy_channels[ch]:
                    quality_score *= 0.5
                
                result['channel_quality'].append(quality_score)
            
            # Overall data validity
            overall_quality = np.mean(result['channel_quality'])
            if overall_quality < 0.3:
                result['valid'] = False
                result['warnings'].append('Overall data quality too low')
            
            result['metrics']['overall_quality'] = overall_quality
            
        except Exception as e:
            result['valid'] = False
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def validate_joint_angles(angles: np.ndarray, limits: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Validate robotic arm joint angles.
        
        Checks that joint angles are within safe operating limits and
        represent physically realizable arm configurations.
        
        Parameters
        ----------
        angles : numpy.ndarray
            Joint angles in radians
        limits : tuple of numpy.ndarray, optional
            (min_angles, max_angles) joint limits
            
        Returns
        -------
        validation_result : dict
            Dictionary containing validation results
        """
        result = {
            'valid': True,
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Default joint limits if not provided
            if limits is None:
                num_joints = len(angles)
                min_limits = np.full(num_joints, -np.pi)
                max_limits = np.full(num_joints, np.pi)
                limits = (min_limits, max_limits)
            
            min_limits, max_limits = limits
            
            # Check joint limits
            below_min = angles < min_limits
            above_max = angles > max_limits
            
            if np.any(below_min):
                violating_joints = np.where(below_min)[0].tolist()
                result['warnings'].append(f'Joints below minimum limits: {violating_joints}')
                result['valid'] = False
            
            if np.any(above_max):
                violating_joints = np.where(above_max)[0].tolist()
                result['warnings'].append(f'Joints above maximum limits: {violating_joints}')
                result['valid'] = False
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(angles)):
                result['warnings'].append('Non-finite joint angles detected')
                result['valid'] = False
            
            # Calculate metrics
            result['metrics']['joint_angles'] = angles.tolist()
            result['metrics']['within_limits'] = np.all((angles >= min_limits) & (angles <= max_limits))
            result['metrics']['range_utilization'] = np.abs(angles) / np.maximum(np.abs(max_limits), np.abs(min_limits))
            
        except Exception as e:
            result['valid'] = False
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def validate_workspace_position(position: np.ndarray, workspace_limits: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Validate end-effector position against workspace boundaries.
        
        Parameters
        ----------
        position : numpy.ndarray
            3D position [x, y, z] in meters
        workspace_limits : dict
            Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples
            
        Returns
        -------
        validation_result : dict
            Dictionary containing validation results
        """
        result = {
            'valid': True,
            'warnings': [],
            'metrics': {}
        }
        
        try:
            if len(position) != 3:
                result['valid'] = False
                result['error'] = 'Position must be 3D coordinate'
                return result
            
            x, y, z = position
            
            # Check each axis
            for axis, coord in zip(['x', 'y', 'z'], [x, y, z]):
                if axis in workspace_limits:
                    min_limit, max_limit = workspace_limits[axis]
                    
                    if coord < min_limit:
                        result['warnings'].append(f'{axis.upper()}-coordinate below minimum: {coord} < {min_limit}')
                        result['valid'] = False
                    
                    if coord > max_limit:
                        result['warnings'].append(f'{axis.upper()}-coordinate above maximum: {coord} > {max_limit}')
                        result['valid'] = False
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(position)):
                result['warnings'].append('Non-finite position coordinates')
                result['valid'] = False
            
            # Calculate distance from workspace center
            center = np.array([
                np.mean(workspace_limits['x']),
                np.mean(workspace_limits['y']),
                np.mean(workspace_limits['z'])
            ])
            
            distance_from_center = np.linalg.norm(position - center)
            
            result['metrics']['position'] = position.tolist()
            result['metrics']['distance_from_center'] = distance_from_center
            result['metrics']['within_workspace'] = result['valid']
            
        except Exception as e:
            result['valid'] = False
            result['error'] = str(e)
        
        return result


class Logger:
    """
    Enhanced logging system for the BCI application.
    
    Provides structured logging with multiple output formats, log levels,
    and specialized handlers for different system components.
    
    Includes performance monitoring, error tracking, and debugging
    utilities optimized for real-time BCI applications.
    """
    
    @staticmethod
    def setup_logging(config: BCIConfig) -> logging.Logger:
        """
        Setup logging configuration for the BCI system.
        
        Parameters
        ----------
        config : BCIConfig
            System configuration containing logging parameters
            
        Returns
        -------
        logger : logging.Logger
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger('bci_system')
        logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if config.log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(config.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def create_default_config_file(path: str = "config.yaml") -> None:
    """
    Create a default configuration file.
    
    Generates a YAML configuration file with default parameters and
    comprehensive documentation for each setting.
    
    Parameters
    ----------
    path : str
        Path where to create the configuration file
    """
    config = BCIConfig()
    config_dict = asdict(config)
    
    # Add documentation
    documented_config = {
        '# BCI Robotic Arm Controller Configuration': None,
        '# Authors: Ibrahim Mediouni, Selim Ouirari': None,
        '# Date: July 2022': None,
        '': None,
        
        '# EEG Acquisition Settings': None,
        'eeg_sampling_rate': config_dict['eeg_sampling_rate'],  # Hz
        'eeg_channels': config_dict['eeg_channels'],            # Number of channels
        'eeg_port': config_dict['eeg_port'],                    # Serial port
        'eeg_baud_rate': config_dict['eeg_baud_rate'],          # Baud rate
        'eeg_daisy_mode': config_dict['eeg_daisy_mode'],        # 16-channel mode
        
        '# Signal Processing Settings': None,
        'filter_lowcut': config_dict['filter_lowcut'],          # Hz
        'filter_highcut': config_dict['filter_highcut'],        # Hz
        'notch_frequency': config_dict['notch_frequency'],      # Hz (50/60)
        'notch_quality': config_dict['notch_quality'],          # Q factor
        
        '# Classification Settings': None,
        'model_type': config_dict['model_type'],                # 'rf' or 'svm'
        'feature_scaling': config_dict['feature_scaling'],      # Boolean
        'confidence_threshold': config_dict['confidence_threshold'],  # 0-1
        'classification_window': config_dict['classification_window'],  # seconds
        
        '# Robotic Arm Settings': None,
        'arm_port': config_dict['arm_port'],                    # Serial port
        'arm_baud_rate': config_dict['arm_baud_rate'],          # Baud rate
        'arm_joints': config_dict['arm_joints'],                # Number of joints
        'arm_step_size': config_dict['arm_step_size'],          # meters
        
        '# Safety Settings': None,
        'max_joint_velocity': config_dict['max_joint_velocity'], # rad/s
        'workspace_x_limits': config_dict['workspace_x_limits'], # (min, max) meters
        'workspace_y_limits': config_dict['workspace_y_limits'], # (min, max) meters
        'workspace_z_limits': config_dict['workspace_z_limits'], # (min, max) meters
        'emergency_stop_enabled': config_dict['emergency_stop_enabled'],
        
        '# System Settings': None,
        'log_level': config_dict['log_level'],                  # DEBUG, INFO, WARNING, ERROR
        'log_file': config_dict['log_file'],                    # Optional log file path
    }
    
    with open(path, 'w') as f:
        for key, value in documented_config.items():
            if key.startswith('#') or key == '':
                f.write(f"{key}\n")
            elif value is not None:
                f.write(f"{key}: {value}\n")
    
    print(f"Default configuration file created: {path}")


def main():
    """
    Main function for utility testing and configuration management.
    
    Provides command-line interface for configuration file generation
    and utility function testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='BCI System Utilities')
    parser.add_argument('--create-config', help='Create default config file')
    parser.add_argument('--validate-config', help='Validate config file')
    parser.add_argument('--test-validation', action='store_true', help='Test data validation')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config_file(args.create_config)
    
    elif args.validate_config:
        try:
            config_manager = ConfigManager(args.validate_config)
            print(f"Configuration valid: {args.validate_config}")
            print(f"EEG Channels: {config_manager.config.eeg_channels}")
            print(f"Arm Joints: {config_manager.config.arm_joints}")
        except Exception as e:
            print(f"Configuration invalid: {e}")
    
    elif args.test_validation:
        # Test EEG data validation
        print("Testing EEG data validation...")
        test_data = np.random.randn(1000, 8) * 10  # 1000 samples, 8 channels
        validation_result = DataValidator.validate_eeg_data(test_data)
        print(f"EEG validation result: {validation_result['valid']}")
        print(f"Quality scores: {validation_result['channel_quality']}")
        
        # Test joint angle validation
        print("\nTesting joint angle validation...")
        test_angles = np.array([0.1, -0.5, 0.8, -0.2, 0.3, -0.1])
        joint_result = DataValidator.validate_joint_angles(test_angles)
        print(f"Joint validation result: {joint_result['valid']}")
        
        # Test workspace validation
        print("\nTesting workspace validation...")
        test_position = np.array([0.3, 0.1, 0.2])
        workspace_limits = {'x': (0.1, 0.6), 'y': (-0.4, 0.4), 'z': (0.0, 0.5)}
        workspace_result = DataValidator.validate_workspace_position(test_position, workspace_limits)
        print(f"Workspace validation result: {workspace_result['valid']}")
    
    else:
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()
# Updated on 2022-07-13
# Updated on 2022-08-18
# Updated on 2022-09-26
# Updated on 2022-10-30
# Updated on 2022-11-16
# Updated on 2023-02-20
# Updated on 2023-03-06
# Updated on 2023-03-07
# Updated on 2023-04-12
# Updated on 2023-05-18
# Updated on 2023-05-25
# Updated on 2023-06-13
# Updated on 2023-06-14
# Updated on 2023-06-12