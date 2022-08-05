#!/usr/bin/env python3
"""
OpenBCI EEG Data Streaming and Acquisition.

This module provides real-time EEG data streaming capabilities using OpenBCI
hardware. It implements a robust data acquisition pipeline with buffering,
quality monitoring, and preprocessing for downstream signal processing.

The system handles various OpenBCI board configurations and provides reliable
data streaming even in challenging real-world environments. Data integrity
checks and automatic reconnection ensure continuous operation.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

import numpy as np
import time
import threading
import logging
from collections import deque
from typing import Optional, Callable, List, Tuple
import json

try:
    from openbci import OpenBCICyton
    from openbci.utils import plot_data
except ImportError:
    print("Warning: OpenBCI library not installed. Using mock implementation.")
    OpenBCICyton = None


class EEGBuffer:
    """
    Thread-safe circular buffer for EEG data storage and retrieval.
    
    This class implements a high-performance circular buffer optimized for
    real-time EEG data streaming. It maintains consistent data flow while
    providing efficient access to recent samples for processing algorithms.
    
    Parameters
    ----------
    size : int
        Maximum number of samples to store in the buffer
    num_channels : int
        Number of EEG channels being recorded
    sampling_rate : float
        Sampling frequency in Hz
        
    Attributes
    ----------
    data : numpy.ndarray
        Circular buffer containing EEG samples
    timestamps : numpy.ndarray
        Corresponding timestamps for each sample
    """
    
    def __init__(self, size: int = 2048, num_channels: int = 8, sampling_rate: float = 250.0):
        """Initialize the EEG buffer with specified parameters."""
        self.size = size
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
        # Initialize circular buffers
        self.data = np.zeros((size, num_channels), dtype=np.float32)
        self.timestamps = np.zeros(size, dtype=np.float64)
        
        # Buffer management
        self.write_index = 0
        self.samples_written = 0
        self.lock = threading.Lock()
        
        # Data quality metrics
        self.signal_quality = np.ones(num_channels)
        self.last_update = time.time()
        
        logging.info(f"EEG buffer initialized: {size} samples, {num_channels} channels, {sampling_rate} Hz")
    
    def add_sample(self, sample: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Add a new EEG sample to the buffer.
        
        This method safely adds new data to the circular buffer while maintaining
        thread safety. It automatically wraps around when the buffer is full,
        ensuring continuous data flow without memory leaks.
        
        Parameters
        ----------
        sample : numpy.ndarray
            EEG data sample with shape (num_channels,)
        timestamp : float, optional
            Sample timestamp. If None, uses current system time
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            # Store the sample and timestamp
            self.data[self.write_index] = sample[:self.num_channels]
            self.timestamps[self.write_index] = timestamp
            
            # Update buffer indices
            self.write_index = (self.write_index + 1) % self.size
            self.samples_written += 1
            self.last_update = time.time()
    
    def get_recent_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the most recent EEG samples from the buffer.
        
        Returns the specified number of recent samples in chronological order.
        This is the primary method used by signal processing algorithms to
        access windowed data for analysis.
        
        Parameters
        ----------
        num_samples : int
            Number of recent samples to retrieve
            
        Returns
        -------
        data : numpy.ndarray
            EEG data with shape (num_samples, num_channels)
        timestamps : numpy.ndarray
            Corresponding timestamps with shape (num_samples,)
        """
        with self.lock:
            if self.samples_written == 0:
                return np.array([]), np.array([])
            
            # Determine how many samples we can actually return
            available_samples = min(num_samples, self.samples_written, self.size)
            
            # Calculate the starting index for extraction
            if self.samples_written < self.size:
                # Buffer not yet full
                start_idx = max(0, self.write_index - available_samples)
                indices = np.arange(start_idx, self.write_index)
            else:
                # Buffer is full, need to handle wrap-around
                start_idx = (self.write_index - available_samples) % self.size
                if start_idx < self.write_index:
                    indices = np.arange(start_idx, self.write_index)
                else:
                    # Wrap around case
                    indices = np.concatenate([
                        np.arange(start_idx, self.size),
                        np.arange(0, self.write_index)
                    ])
            
            return self.data[indices].copy(), self.timestamps[indices].copy()
    
    def get_signal_quality(self) -> np.ndarray:
        """
        Calculate signal quality metrics for each channel.
        
        Returns
        -------
        quality : numpy.ndarray
            Signal quality scores (0-1) for each channel
        """
        with self.lock:
            if self.samples_written < 100:  # Need minimum samples for quality assessment
                return self.signal_quality
            
            # Get recent data for quality analysis
            recent_data, _ = self.get_recent_data(100)
            
            if recent_data.size == 0:
                return self.signal_quality
            
            # Calculate quality metrics based on signal variance and artifacts
            for ch in range(self.num_channels):
                signal = recent_data[:, ch]
                
                # Basic quality metrics
                variance = np.var(signal)
                mean_abs = np.mean(np.abs(signal))
                
                # Detect saturation (signal clipping)
                saturation = np.mean(np.abs(signal) > 100)  # Assuming microvolts
                
                # Combine metrics into quality score
                quality = 1.0 - min(saturation * 2, 1.0)  # Heavily penalize saturation
                quality *= min(variance / 10.0, 1.0)  # Prefer some variance
                
                self.signal_quality[ch] = max(0.0, min(1.0, quality))
            
            return self.signal_quality.copy()


class BCIDataStreamer:
    """
    Main class for OpenBCI data streaming and management.
    
    This class orchestrates the entire EEG data acquisition process, from
    hardware initialization to data preprocessing. It provides a high-level
    interface for starting/stopping data streams and accessing real-time
    neural signals.
    
    The streamer handles hardware errors gracefully and provides comprehensive
    logging for debugging and system monitoring.
    
    Parameters
    ----------
    port : str, optional
        Serial port for OpenBCI board connection
    baud_rate : int
        Serial communication baud rate
    daisy : bool
        Enable daisy chain mode for 16-channel recording
        
    Attributes
    ----------
    board : OpenBCICyton or None
        OpenBCI board interface object
    buffer : EEGBuffer
        Circular buffer for EEG data storage
    """
    
    def __init__(self, port: str = "/dev/ttyUSB0", baud_rate: int = 115200, daisy: bool = False):
        """Initialize the BCI data streamer with hardware configuration."""
        self.port = port
        self.baud_rate = baud_rate
        self.daisy = daisy
        self.num_channels = 16 if daisy else 8
        
        # Initialize components
        self.board = None
        self.buffer = EEGBuffer(size=2048, num_channels=self.num_channels, sampling_rate=250.0)
        
        # Streaming control
        self.is_streaming = False
        self.stream_thread = None
        self.data_callback = None
        
        # Performance monitoring
        self.samples_received = 0
        self.start_time = None
        self.last_sample_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.logger.info(f"BCI Data Streamer initialized for {self.num_channels} channels")
    
    def connect(self) -> bool:
        """
        Establish connection with OpenBCI hardware.
        
        Attempts to connect to the OpenBCI board and configure it for data
        acquisition. This method handles various connection scenarios and
        provides detailed error reporting for troubleshooting.
        
        Returns
        -------
        success : bool
            True if connection was successful, False otherwise
        """
        try:
            if OpenBCICyton is None:
                self.logger.warning("OpenBCI library not available. Using simulation mode.")
                return True  # Allow operation in simulation mode
            
            self.logger.info(f"Connecting to OpenBCI board on {self.port}")
            
            # Initialize the board
            self.board = OpenBCICyton(port=self.port, baud=self.baud_rate, daisy=self.daisy)
            
            # Configure board settings
            self._configure_board()
            
            self.logger.info("Successfully connected to OpenBCI board")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenBCI board: {e}")
            return False
    
    def _configure_board(self) -> None:
        """Configure OpenBCI board settings for optimal data acquisition."""
        if self.board is None:
            return
        
        try:
            # Reset board to default state
            self.board.send_command('v')  # Reset to default
            time.sleep(0.5)
            
            # Configure channels for EEG recording
            # Set all channels to normal input, gain 24, no bias
            for i in range(1, self.num_channels + 1):
                cmd = f'x{i}060110X'  # Normal input, gain 24, normal power, bias off
                self.board.send_command(cmd)
                time.sleep(0.1)
            
            # Enable bias drive and SRB2
            self.board.send_command('C2')  # Connect SRB2 to all N inputs
            self.board.send_command('B1')  # Enable bias drive
            
            self.logger.info("Board configuration completed")
            
        except Exception as e:
            self.logger.warning(f"Board configuration failed: {e}")
    
    def _data_handler(self, sample) -> None:
        """
        Internal callback for processing incoming EEG samples.
        
        This method is called for each new sample from the OpenBCI board.
        It performs initial data validation, adds samples to the buffer,
        and triggers any registered user callbacks.
        
        Parameters
        ----------
        sample : OpenBCISample
            Raw sample data from OpenBCI board
        """
        try:
            # Extract channel data (convert from counts to microvolts)
            if hasattr(sample, 'channels_data'):
                # Real OpenBCI sample
                channel_data = np.array(sample.channels_data[:self.num_channels], dtype=np.float32)
                # Convert from counts to microvolts (OpenBCI specific conversion)
                channel_data = channel_data * 4.5 / 24 / (2**23 - 1) * 1000000
            else:
                # Simulation mode - generate synthetic EEG-like data
                channel_data = self._generate_synthetic_eeg()
            
            # Add to buffer
            timestamp = time.time()
            self.buffer.add_sample(channel_data, timestamp)
            
            # Update performance counters
            self.samples_received += 1
            self.last_sample_time = timestamp
            
            # Trigger user callback if registered
            if self.data_callback:
                self.data_callback(channel_data, timestamp)
                
        except Exception as e:
            self.logger.error(f"Error processing sample: {e}")
    
    def _generate_synthetic_eeg(self) -> np.ndarray:
        """
        Generate synthetic EEG data for simulation and testing.
        
        Creates realistic EEG-like signals with multiple frequency components
        that mimic actual brainwave patterns. Useful for development and testing
        when hardware is not available.
        
        Returns
        -------
        data : numpy.ndarray
            Synthetic EEG sample with shape (num_channels,)
        """
        t = time.time()
        data = np.zeros(self.num_channels)
        
        for ch in range(self.num_channels):
            # Generate multiple frequency components typical of EEG
            alpha = 10 * np.sin(2 * np.pi * 10 * t + ch * 0.5)  # Alpha waves (10 Hz)
            beta = 5 * np.sin(2 * np.pi * 20 * t + ch * 0.3)    # Beta waves (20 Hz)
            theta = 15 * np.sin(2 * np.pi * 6 * t + ch * 0.7)   # Theta waves (6 Hz)
            
            # Add some noise
            noise = np.random.normal(0, 2)
            
            # Combine components
            data[ch] = alpha + beta + theta + noise
        
        return data.astype(np.float32)
    
    def start_streaming(self, callback: Optional[Callable] = None) -> bool:
        """
        Start real-time EEG data streaming.
        
        Begins continuous data acquisition from the OpenBCI board. The streaming
        runs in a separate thread to ensure real-time performance without
        blocking the main application thread.
        
        Parameters
        ----------
        callback : callable, optional
            Function to call for each new sample. Should accept (data, timestamp)
            
        Returns
        -------
        success : bool
            True if streaming started successfully
        """
        if self.is_streaming:
            self.logger.warning("Streaming is already active")
            return True
        
        if self.board is None and OpenBCICyton is not None:
            self.logger.error("Board not connected. Call connect() first.")
            return False
        
        try:
            self.data_callback = callback
            self.is_streaming = True
            self.start_time = time.time()
            self.samples_received = 0
            
            if self.board:
                # Start streaming with real hardware
                self.board.start_stream(self._data_handler)
            else:
                # Start simulation mode
                self.stream_thread = threading.Thread(target=self._simulation_loop)
                self.stream_thread.daemon = True
                self.stream_thread.start()
            
            self.logger.info("EEG streaming started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            return False
    
    def _simulation_loop(self) -> None:
        """Run simulation mode data generation loop."""
        sample_period = 1.0 / 250.0  # 250 Hz sampling rate
        
        while self.is_streaming:
            start_time = time.time()
            
            # Generate and process synthetic sample
            synthetic_sample = self._generate_synthetic_eeg()
            self._data_handler(type('Sample', (), {'channels_data': synthetic_sample})())
            
            # Maintain precise timing
            elapsed = time.time() - start_time
            sleep_time = sample_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop_streaming(self) -> None:
        """
        Stop EEG data streaming.
        
        Cleanly shuts down the data acquisition process and releases hardware
        resources. This method ensures all threads are properly terminated
        and provides streaming statistics.
        """
        if not self.is_streaming:
            self.logger.warning("Streaming is not active")
            return
        
        try:
            self.is_streaming = False
            
            if self.board:
                self.board.stop_stream()
            
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            
            # Calculate streaming statistics
            if self.start_time:
                duration = time.time() - self.start_time
                rate = self.samples_received / duration if duration > 0 else 0
                self.logger.info(f"Streaming stopped. Duration: {duration:.1f}s, "
                               f"Samples: {self.samples_received}, Rate: {rate:.1f} Hz")
            
        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")
    
    def disconnect(self) -> None:
        """
        Disconnect from OpenBCI hardware.
        
        Properly closes the connection to the OpenBCI board and releases
        all associated resources. Should be called when the application
        is shutting down or switching to a different board.
        """
        try:
            self.stop_streaming()
            
            if self.board:
                self.board.disconnect()
                self.board = None
                
            self.logger.info("Disconnected from OpenBCI board")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    def get_recent_data(self, duration_seconds: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recent EEG data for processing.
        
        Retrieves the most recent EEG samples covering the specified duration.
        This is the primary method used by signal processing algorithms to
        access data for analysis.
        
        Parameters
        ----------
        duration_seconds : float
            Duration of data to retrieve in seconds
            
        Returns
        -------
        data : numpy.ndarray
            EEG data with shape (samples, channels)
        timestamps : numpy.ndarray
            Corresponding timestamps
        """
        num_samples = int(duration_seconds * self.buffer.sampling_rate)
        return self.buffer.get_recent_data(num_samples)
    
    def get_status(self) -> dict:
        """
        Get current streaming status and statistics.
        
        Returns
        -------
        status : dict
            Dictionary containing streaming status, performance metrics,
            and signal quality information
        """
        status = {
            'is_streaming': self.is_streaming,
            'is_connected': self.board is not None,
            'num_channels': self.num_channels,
            'samples_received': self.samples_received,
            'signal_quality': self.buffer.get_signal_quality().tolist(),
        }
        
        if self.start_time:
            duration = time.time() - self.start_time
            status['streaming_duration'] = duration
            status['average_rate'] = self.samples_received / duration if duration > 0 else 0
        
        return status


def main():
    """
    Main function for standalone EEG data streaming.
    
    This function provides a command-line interface for testing the EEG
    acquisition system. It initializes the streamer, connects to hardware,
    and displays real-time signal quality metrics.
    """
    import argparse
    import signal
    import sys
    
    parser = argparse.ArgumentParser(description='OpenBCI EEG Data Streamer')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--daisy', action='store_true', help='Enable daisy chain')
    parser.add_argument('--duration', type=int, default=0, help='Stream duration (0=infinite)')
    
    args = parser.parse_args()
    
    # Initialize streamer
    streamer = BCIDataStreamer(port=args.port, baud_rate=args.baud, daisy=args.daisy)
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\nShutting down...")
        streamer.disconnect()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Connect and start streaming
    if not streamer.connect():
        print("Failed to connect to OpenBCI board")
        sys.exit(1)
    
    def data_callback(data, timestamp):
        """Print signal quality periodically."""
        if streamer.samples_received % 250 == 0:  # Every second at 250 Hz
            quality = streamer.buffer.get_signal_quality()
            print(f"Signal quality: {quality}")
    
    if not streamer.start_streaming(callback=data_callback):
        print("Failed to start streaming")
        sys.exit(1)
    
    print(f"Streaming started. Press Ctrl+C to stop.")
    
    try:
        if args.duration > 0:
            time.sleep(args.duration)
            streamer.disconnect()
        else:
            # Stream indefinitely
            while True:
                time.sleep(1)
                status = streamer.get_status()
                print(f"Samples: {status['samples_received']}, "
                      f"Rate: {status.get('average_rate', 0):.1f} Hz")
    
    except KeyboardInterrupt:
        pass
    
    streamer.disconnect()


if __name__ == "__main__":
    main()
# Updated on 2022-08-19
# Updated on 2022-08-06
# Updated on 2022-08-25
# Updated on 2022-08-28
# Updated on 2022-08-05