#!/usr/bin/env python3
"""
Main Application Entry Point for BCI Robotic Arm Controller.

This is the primary entry point for the brain-computer interface robotic arm
control system. It provides a unified interface to initialize, configure,
and run all system components including EEG acquisition, signal processing,
brain state classification, and robotic arm control.

The application supports multiple operation modes including real-time control,
training data collection, model training, and system testing.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

import sys
import os
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import BCI system components
from openbci_reader.stream_data import BCIDataStreamer
from signal_processing.eeg_classifier import EEGClassifier, SignalProcessor
from ros2_publisher.arm_control_node import ArmControlNode, RobotController
from utils import ConfigManager, Logger, BCIConfig

# Check for ROS2 availability
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Some features will be limited.")


class BCIApplication:
    """
    Main BCI application controller.
    
    This class orchestrates the entire BCI system, managing component
    initialization, inter-component communication, and system lifecycle.
    It provides a high-level interface for different operation modes
    and handles system shutdown gracefully.
    
    Parameters
    ----------
    config_path : str, optional
        Path to system configuration file
        
    Attributes
    ----------
    config : BCIConfig
        System configuration
    components : dict
        Dictionary of initialized system components
    is_running : bool
        System running state
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the BCI application."""
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self.logger = Logger.setup_logging(self.config)
        self.logger.info("BCI Application initializing...")
        
        # Initialize component containers
        self.components = {}
        self.is_running = False
        self.ros_node = None
        
        # Performance monitoring
        self.start_time = None
        self.performance_stats = {
            'uptime': 0,
            'commands_processed': 0,
            'errors_encountered': 0,
            'average_processing_time': 0
        }
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("BCI Application initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def initialize_components(self, mode: str = "full") -> bool:
        """
        Initialize system components based on operation mode.
        
        Parameters
        ----------
        mode : str
            Initialization mode: 'full', 'eeg_only', 'arm_only', 'processing_only'
            
        Returns
        -------
        success : bool
            True if all required components initialized successfully
        """
        try:
            self.logger.info(f"Initializing components for mode: {mode}")
            
            # Initialize EEG components
            if mode in ['full', 'eeg_only', 'processing_only']:
                self.logger.info("Initializing EEG data streamer...")
                self.components['eeg_streamer'] = BCIDataStreamer(
                    port=self.config.eeg_port,
                    baud_rate=self.config.eeg_baud_rate,
                    daisy=self.config.eeg_daisy_mode
                )
                
                self.logger.info("Initializing signal processor...")
                self.components['signal_processor'] = SignalProcessor(
                    sampling_rate=self.config.eeg_sampling_rate,
                    num_channels=self.config.eeg_channels
                )
                
                self.logger.info("Initializing EEG classifier...")
                self.components['eeg_classifier'] = EEGClassifier(
                    model_type=self.config.model_type,
                    feature_scaler=self.config.feature_scaling
                )
            
            # Initialize robotic arm components
            if mode in ['full', 'arm_only']:
                self.logger.info("Initializing robot controller...")
                self.components['robot_controller'] = RobotController(
                    port=self.config.arm_port,
                    baud_rate=self.config.arm_baud_rate,
                    num_joints=self.config.arm_joints
                )
            
            # Initialize ROS2 node for full system
            if mode == 'full' and ROS2_AVAILABLE:
                self.logger.info("Initializing ROS2 node...")
                rclpy.init()
                self.ros_node = ArmControlNode()
                self.components['ros_node'] = self.ros_node
            
            # Connect to hardware
            self._connect_hardware()
            
            # Load trained models if available
            self._load_models()
            
            self.logger.info("Component initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False
    
    def _connect_hardware(self) -> None:
        """Connect to all hardware components."""
        # Connect EEG hardware
        if 'eeg_streamer' in self.components:
            if self.components['eeg_streamer'].connect():
                self.logger.info("EEG hardware connected")
            else:
                self.logger.warning("EEG hardware connection failed, using simulation")
        
        # Connect robotic arm
        if 'robot_controller' in self.components:
            if self.components['robot_controller'].connect():
                self.logger.info("Robotic arm connected")
            else:
                self.logger.warning("Robotic arm connection failed, using simulation")
    
    def _load_models(self) -> None:
        """Load trained machine learning models."""
        if 'eeg_classifier' in self.components:
            model_path = "models/eeg_model.pkl"  # Should be configurable
            if os.path.exists(model_path):
                if self.components['eeg_classifier'].load_model(model_path):
                    self.logger.info(f"Loaded EEG model from {model_path}")
                else:
                    self.logger.warning("Failed to load EEG model")
            else:
                self.logger.warning("No trained EEG model found")
    
    def run_realtime_control(self) -> None:
        """
        Run real-time brain-controlled robotic arm operation.
        
        This is the main operation mode where EEG signals are continuously
        processed and classified to control the robotic arm in real-time.
        """
        if not self._validate_realtime_requirements():
            return
        
        self.logger.info("Starting real-time BCI control...")
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Start EEG data streaming
            eeg_streamer = self.components['eeg_streamer']
            if not eeg_streamer.start_streaming(self._eeg_data_callback):
                self.logger.error("Failed to start EEG streaming")
                return
            
            # Run ROS2 node if available
            if self.ros_node:
                if not self.ros_node.initialize_system():
                    self.logger.error("Failed to initialize ROS2 system")
                    return
                
                self.logger.info("Running ROS2 control loop...")
                rclpy.spin(self.ros_node)
            else:
                # Run non-ROS2 control loop
                self._run_control_loop()
                
        except KeyboardInterrupt:
            self.logger.info("Control loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Control loop error: {e}")
        finally:
            self.is_running = False
    
    def _validate_realtime_requirements(self) -> bool:
        """Validate that all required components are available for real-time control."""
        required_components = ['eeg_streamer', 'signal_processor', 'eeg_classifier', 'robot_controller']
        
        for component in required_components:
            if component not in self.components:
                self.logger.error(f"Required component missing: {component}")
                return False
        
        # Check if classifier is trained
        if not self.components['eeg_classifier'].is_trained:
            self.logger.error("EEG classifier not trained. Train a model first.")
            return False
        
        return True
    
    def _eeg_data_callback(self, data, timestamp):
        """Callback for processing incoming EEG data."""
        if not self.is_running:
            return
        
        try:
            # This callback is called for each EEG sample
            # For real-time processing, we'll process data periodically
            pass  # Processing happens in the main control loop
            
        except Exception as e:
            self.logger.error(f"EEG data callback error: {e}")
            self.performance_stats['errors_encountered'] += 1
    
    def _run_control_loop(self) -> None:
        """Run the main control loop without ROS2."""
        self.logger.info("Running non-ROS2 control loop...")
        
        processing_interval = 0.2  # Process every 200ms
        last_processing_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process EEG data periodically
                if current_time - last_processing_time >= processing_interval:
                    self._process_brain_signals()
                    last_processing_time = current_time
                
                # Update performance stats
                self.performance_stats['uptime'] = current_time - self.start_time
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                self.performance_stats['errors_encountered'] += 1
                time.sleep(0.1)  # Longer sleep on error
    
    def _process_brain_signals(self) -> None:
        """Process EEG signals and execute arm commands."""
        try:
            # Get recent EEG data
            eeg_streamer = self.components['eeg_streamer']
            signal_processor = self.components['signal_processor']
            eeg_classifier = self.components['eeg_classifier']
            robot_controller = self.components['robot_controller']
            
            # Get recent data for analysis
            eeg_data, timestamps = eeg_streamer.get_recent_data(
                duration_seconds=self.config.classification_window
            )
            
            if eeg_data.size == 0:
                return
            
            # Process the signal
            start_time = time.time()
            features = signal_processor.process_segment(eeg_data)
            
            if not features:
                return
            
            # Classify brain state
            prediction, confidence, probabilities = eeg_classifier.predict(features)
            
            # Execute command if confidence is high enough
            if confidence >= self.config.confidence_threshold:
                # Map brain state to arm command
                command_mapping = {
                    'rest': 'STOP',
                    'focus': 'MOVE_FORWARD',
                    'relax': 'MOVE_BACKWARD',
                    'movement': 'MOVE_UP',
                    'left_hand': 'MOVE_LEFT',
                    'right_hand': 'MOVE_RIGHT',
                }
                
                if prediction in command_mapping:
                    # Convert to arm command enum and execute
                    from ros2_publisher.arm_control_node import ArmCommand
                    arm_command = getattr(ArmCommand, command_mapping[prediction])
                    
                    if robot_controller.execute_command(arm_command):
                        self.performance_stats['commands_processed'] += 1
                        self.logger.debug(f"Executed: {prediction} -> {arm_command.value} "
                                        f"(confidence: {confidence:.3f})")
            
            # Update processing time statistics
            processing_time = time.time() - start_time
            self.performance_stats['average_processing_time'] = (
                self.performance_stats['average_processing_time'] * 0.9 + 
                processing_time * 0.1
            )
            
        except Exception as e:
            self.logger.error(f"Brain signal processing error: {e}")
            self.performance_stats['errors_encountered'] += 1
    
    def run_training_mode(self, training_data_path: str) -> None:
        """
        Run model training mode.
        
        Parameters
        ----------
        training_data_path : str
            Path to training data file or directory
        """
        self.logger.info(f"Starting training mode with data: {training_data_path}")
        
        if 'eeg_classifier' not in self.components:
            self.logger.error("EEG classifier not initialized")
            return
        
        # TODO: Implement training data loading and model training
        self.logger.info("Training mode not yet implemented")
    
    def run_data_collection(self, output_path: str, duration: float = 300.0) -> None:
        """
        Run data collection mode for training data acquisition.
        
        Parameters
        ----------
        output_path : str
            Path to save collected data
        duration : float
            Collection duration in seconds
        """
        self.logger.info(f"Starting data collection for {duration} seconds...")
        
        if 'eeg_streamer' not in self.components:
            self.logger.error("EEG streamer not initialized")
            return
        
        # TODO: Implement data collection with user prompts
        self.logger.info("Data collection mode not yet implemented")
    
    def run_test_mode(self) -> None:
        """Run system test mode to verify all components."""
        self.logger.info("Starting system test mode...")
        
        # Test EEG streaming
        if 'eeg_streamer' in self.components:
            self.logger.info("Testing EEG streaming...")
            eeg_streamer = self.components['eeg_streamer']
            
            if eeg_streamer.start_streaming():
                time.sleep(5.0)  # Stream for 5 seconds
                status = eeg_streamer.get_status()
                self.logger.info(f"EEG test result: {status['samples_received']} samples, "
                               f"{status.get('average_rate', 0):.1f} Hz")
                eeg_streamer.stop_streaming()
            else:
                self.logger.error("EEG streaming test failed")
        
        # Test signal processing
        if 'signal_processor' in self.components:
            self.logger.info("Testing signal processing...")
            # TODO: Add signal processing tests
        
        # Test robotic arm
        if 'robot_controller' in self.components:
            self.logger.info("Testing robotic arm...")
            robot_controller = self.components['robot_controller']
            
            # Test basic movements
            from ros2_publisher.arm_control_node import ArmCommand
            test_commands = [
                ArmCommand.HOME_POSITION,
                ArmCommand.MOVE_UP,
                ArmCommand.MOVE_DOWN,
                ArmCommand.STOP
            ]
            
            for cmd in test_commands:
                self.logger.info(f"Testing command: {cmd.value}")
                if robot_controller.execute_command(cmd):
                    time.sleep(1.0)  # Wait for execution
                else:
                    self.logger.warning(f"Command failed: {cmd.value}")
        
        self.logger.info("System test complete")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns
        -------
        status : dict
            System status information
        """
        status = {
            'is_running': self.is_running,
            'uptime': self.performance_stats['uptime'],
            'components': list(self.components.keys()),
            'performance': self.performance_stats.copy()
        }
        
        # Add component-specific status
        if 'eeg_streamer' in self.components:
            status['eeg_status'] = self.components['eeg_streamer'].get_status()
        
        if 'robot_controller' in self.components:
            arm_state = self.components['robot_controller'].get_state()
            status['arm_status'] = {
                'connected': self.components['robot_controller'].is_connected,
                'position': arm_state.end_effector_position.tolist(),
                'is_moving': arm_state.is_moving,
                'error_state': arm_state.error_state
            }
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown the BCI application gracefully."""
        self.logger.info("Shutting down BCI application...")
        self.is_running = False
        
        try:
            # Stop EEG streaming
            if 'eeg_streamer' in self.components:
                self.components['eeg_streamer'].disconnect()
            
            # Disconnect robotic arm
            if 'robot_controller' in self.components:
                self.components['robot_controller'].disconnect()
            
            # Shutdown ROS2 node
            if self.ros_node:
                self.ros_node.shutdown()
                self.ros_node.destroy_node()
            
            if ROS2_AVAILABLE and rclpy.ok():
                rclpy.shutdown()
            
            # Log final statistics
            if self.start_time:
                total_uptime = time.time() - self.start_time
                self.logger.info(f"Session statistics:")
                self.logger.info(f"  Total uptime: {total_uptime:.1f} seconds")
                self.logger.info(f"  Commands processed: {self.performance_stats['commands_processed']}")
                self.logger.info(f"  Errors encountered: {self.performance_stats['errors_encountered']}")
                self.logger.info(f"  Average processing time: {self.performance_stats['average_processing_time']:.3f}s")
            
            self.logger.info("BCI application shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """
    Main function for the BCI application.
    
    Provides command-line interface for different operation modes
    and system configuration options.
    """
    parser = argparse.ArgumentParser(
        description='BCI Robotic Arm Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode realtime                    # Run real-time control
  python main.py --mode test                        # Run system tests
  python main.py --mode train --data ./training/    # Train models
  python main.py --mode collect --output ./data/    # Collect training data
  python main.py --config custom_config.yaml       # Use custom config
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['realtime', 'train', 'collect', 'test'],
                       default='realtime',
                       help='Operation mode')
    
    parser.add_argument('--config',
                       help='Configuration file path')
    
    parser.add_argument('--data',
                       help='Training data path (for train mode)')
    
    parser.add_argument('--output',
                       help='Output path (for collect mode)')
    
    parser.add_argument('--duration', type=float, default=300.0,
                       help='Duration in seconds (for collect mode)')
    
    parser.add_argument('--components',
                       choices=['full', 'eeg_only', 'arm_only', 'processing_only'],
                       default='full',
                       help='Components to initialize')
    
    args = parser.parse_args()
    
    # Create and initialize application
    try:
        app = BCIApplication(config_path=args.config)
        
        if not app.initialize_components(mode=args.components):
            print("Failed to initialize components")
            sys.exit(1)
        
        # Run requested mode
        if args.mode == 'realtime':
            app.run_realtime_control()
        elif args.mode == 'train':
            if not args.data:
                print("Training data path required for train mode")
                sys.exit(1)
            app.run_training_mode(args.data)
        elif args.mode == 'collect':
            if not args.output:
                print("Output path required for collect mode")
                sys.exit(1)
            app.run_data_collection(args.output, args.duration)
        elif args.mode == 'test':
            app.run_test_mode()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
    finally:
        if 'app' in locals():
            app.shutdown()


if __name__ == "__main__":
    main()
# Updated on 2022-07-09
# Updated on 2022-08-07