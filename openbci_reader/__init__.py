"""
OpenBCI EEG Data Acquisition Module.

This module handles real-time acquisition of EEG signals from OpenBCI devices.
It provides classes and functions for streaming, buffering, and preprocessing
neural data before feeding it into the signal processing pipeline.

The module supports various OpenBCI hardware configurations and provides
robust error handling for real-world deployment scenarios.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

from .stream_data import BCIDataStreamer, EEGBuffer

__all__ = [
    "BCIDataStreamer",
    "EEGBuffer",
]
