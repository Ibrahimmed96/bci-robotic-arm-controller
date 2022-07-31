"""
EEG Signal Processing and Classification Module.

This module contains algorithms for processing raw EEG signals and classifying
brain states. It implements digital filters, feature extraction methods, and
machine learning classifiers optimized for real-time brain-computer interfaces.

The processing pipeline transforms noisy neural signals into clean, actionable
commands that can be interpreted by robotic control systems.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

from .eeg_classifier import EEGClassifier, SignalProcessor

__all__ = [
    "EEGClassifier",
    "SignalProcessor",
]