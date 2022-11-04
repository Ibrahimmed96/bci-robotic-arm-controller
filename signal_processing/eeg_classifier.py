#!/usr/bin/env python3
"""
EEG Signal Processing and Brain State Classification.

This module implements advanced signal processing algorithms for EEG data analysis
and real-time classification of brain states. It provides a complete pipeline
from raw neural signals to actionable commands for robotic control.

The processing includes digital filtering, artifact removal, feature extraction,
and machine learning classification optimized for brain-computer interfaces.
The system is designed to distinguish between different mental states such as
focus, relaxation, and motor imagery.

Authors: Ibrahim Mediouni, Selim Ouirari
Date: July 2022
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import pickle
import os

# Scientific computing and signal processing
from scipy import signal
from scipy.fft import fft, fftfreq
import scipy.stats as stats

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Data handling
import pandas as pd

# Import our EEG acquisition module
from ..openbci_reader.stream_data import BCIDataStreamer


class SignalProcessor:
    """
    Advanced EEG signal processing pipeline.
    
    This class implements a comprehensive signal processing chain for EEG data,
    including filtering, artifact removal, and feature extraction. The pipeline
    is optimized for real-time operation while maintaining high signal quality.
    
    The processor handles common EEG artifacts like eye blinks, muscle activity,
    and power line interference through sophisticated filtering techniques.
    
    Parameters
    ----------
    sampling_rate : float
        EEG sampling frequency in Hz
    num_channels : int
        Number of EEG channels to process
    filter_config : dict, optional
        Configuration for digital filters
        
    Attributes
    ----------
    filters : dict
        Dictionary of configured digital filters
    artifact_detector : ArtifactDetector
        System for detecting and handling EEG artifacts
    """
    
    def __init__(self, sampling_rate: float = 250.0, num_channels: int = 8, 
                 filter_config: Optional[Dict] = None):
        """Initialize the signal processor with specified parameters."""
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.nyquist = sampling_rate / 2.0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize filters
        self.filters = self._setup_filters(filter_config or {})
        
        # Feature extraction parameters
        self.frequency_bands = {
            'delta': (0.5, 4),      # Deep sleep, unconscious processes
            'theta': (4, 8),        # Drowsiness, meditation, creativity
            'alpha': (8, 13),       # Relaxed awareness, eyes closed
            'beta': (13, 30),       # Active thinking, concentration
            'gamma': (30, 100),     # High-level cognitive processing
        }
        
        # Buffer for continuous processing
        self.processing_buffer = deque(maxlen=int(sampling_rate * 4))  # 4 second buffer
        
        # Artifact detection thresholds
        self.artifact_thresholds = {
            'amplitude': 150.0,     # Microvolts - detect extreme amplitudes
            'gradient': 50.0,       # Rate of change threshold
            'kurtosis': 5.0,        # Statistical measure of signal spikiness
        }
        
        self.logger.info(f"Signal processor initialized: {sampling_rate} Hz, {num_channels} channels")
    
    def _setup_filters(self, config: Dict) -> Dict:
        """
        Initialize digital filters for EEG processing.
        
        Creates a set of optimized digital filters for different aspects of
        EEG signal conditioning. The filters are designed to preserve neural
        information while removing artifacts and noise.
        
        Parameters
        ----------
        config : dict
            Filter configuration parameters
            
        Returns
        -------
        filters : dict
            Dictionary of configured filter objects
        """
        filters = {}
        
        # Bandpass filter for general EEG signal conditioning
        # This removes DC drift and high-frequency noise while preserving EEG content
        lowcut = config.get('bandpass_low', 1.0)    # Remove DC and very low frequencies
        highcut = config.get('bandpass_high', 45.0)  # Remove high-frequency noise
        
        # Design bandpass filter using Butterworth design for smooth response
        sos_bandpass = signal.butter(4, [lowcut, highcut], btype='band', 
                                   fs=self.sampling_rate, output='sos')
        filters['bandpass'] = sos_bandpass
        
        # Notch filter for power line interference (50/60 Hz)
        notch_freq = config.get('notch_freq', 50.0)  # European power line frequency
        notch_q = config.get('notch_q', 30.0)        # Quality factor for narrow notch
        
        sos_notch = signal.iirnotch(notch_freq, notch_q, fs=self.sampling_rate)
        filters['notch'] = sos_notch
        
        # High-pass filter for removing baseline drift
        highpass_cutoff = config.get('highpass', 0.5)
        sos_highpass = signal.butter(2, highpass_cutoff, btype='high', 
                                   fs=self.sampling_rate, output='sos')
        filters['highpass'] = sos_highpass
        
        # Low-pass anti-aliasing filter
        lowpass_cutoff = config.get('lowpass', 45.0)
        sos_lowpass = signal.butter(4, lowpass_cutoff, btype='low', 
                                  fs=self.sampling_rate, output='sos')
        filters['lowpass'] = sos_lowpass
        
        self.logger.info(f"Filters configured: BP({lowcut}-{highcut}Hz), "
                        f"Notch({notch_freq}Hz), HP({highpass_cutoff}Hz)")
        
        return filters
    
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply digital filters to EEG data.
        
        Processes the input EEG signal through the configured filter chain
        to remove artifacts and noise while preserving neural information.
        The filtering is applied in a specific order to maximize effectiveness.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input EEG data with shape (samples, channels)
            
        Returns
        -------
        filtered_data : numpy.ndarray
            Filtered EEG data with the same shape as input
        """
        if data.size == 0:
            return data
        
        # Ensure data is in the correct format (samples x channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        filtered_data = data.copy()
        
        try:
            # Apply filters in optimal order
            # 1. High-pass to remove baseline drift
            filtered_data = signal.sosfilt(self.filters['highpass'], filtered_data, axis=0)
            
            # 2. Notch filter to remove power line interference
            filtered_data = signal.sosfilt(self.filters['notch'], filtered_data, axis=0)
            
            # 3. Bandpass for general signal conditioning
            filtered_data = signal.sosfilt(self.filters['bandpass'], filtered_data, axis=0)
            
            # 4. Low-pass for anti-aliasing
            filtered_data = signal.sosfilt(self.filters['lowpass'], filtered_data, axis=0)
            
        except Exception as e:
            self.logger.error(f"Filter application failed: {e}")
            return data  # Return original data if filtering fails
        
        return filtered_data
    
    def detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Detect artifacts in EEG signals.
        
        Identifies various types of artifacts including eye blinks, muscle
        activity, electrode pops, and other non-neural signals. The detection
        uses multiple statistical and morphological criteria.
        
        Parameters
        ----------
        data : numpy.ndarray
            EEG data to analyze for artifacts
            
        Returns
        -------
        artifact_mask : numpy.ndarray
            Boolean mask indicating artifact locations (True = artifact)
        """
        if data.size == 0:
            return np.array([], dtype=bool)
        
        # Ensure correct data format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        samples, channels = data.shape
        artifact_mask = np.zeros(samples, dtype=bool)
        
        try:
            # 1. Amplitude-based artifact detection
            # Detect samples with extreme amplitudes (likely electrode artifacts)
            extreme_amplitude = np.any(np.abs(data) > self.artifact_thresholds['amplitude'], axis=1)
            
            # 2. Gradient-based detection (rapid signal changes)
            # Calculate signal derivatives to detect sudden jumps
            if samples > 1:
                gradients = np.diff(data, axis=0)
                extreme_gradient = np.any(np.abs(gradients) > self.artifact_thresholds['gradient'], axis=1)
                extreme_gradient = np.concatenate([[False], extreme_gradient])  # Pad to match length
            else:
                extreme_gradient = np.zeros(samples, dtype=bool)
            
            # 3. Statistical outlier detection using kurtosis
            # High kurtosis indicates spiky, artifact-like signals
            window_size = min(int(self.sampling_rate * 0.5), samples)  # 0.5 second windows
            high_kurtosis = np.zeros(samples, dtype=bool)
            
            for i in range(0, samples - window_size + 1, window_size // 2):
                window_data = data[i:i + window_size]
                if window_data.size > 10:  # Minimum samples for kurtosis calculation
                    for ch in range(channels):
                        kurt = stats.kurtosis(window_data[:, ch])
                        if abs(kurt) > self.artifact_thresholds['kurtosis']:
                            high_kurtosis[i:i + window_size] = True
            
            # Combine artifact detection criteria
            artifact_mask = extreme_amplitude | extreme_gradient | high_kurtosis
            
            # Apply morphological operations to clean up detection
            # Dilate to include samples around detected artifacts
            kernel_size = int(self.sampling_rate * 0.1)  # 100ms dilation
            if kernel_size > 1:
                kernel = np.ones(kernel_size, dtype=bool)
                artifact_mask = np.convolve(artifact_mask.astype(int), kernel, mode='same') > 0
            
        except Exception as e:
            self.logger.error(f"Artifact detection failed: {e}")
            artifact_mask = np.zeros(samples, dtype=bool)
        
        artifact_percentage = np.mean(artifact_mask) * 100
        if artifact_percentage > 5:  # Log if more than 5% artifacts detected
            self.logger.warning(f"High artifact rate detected: {artifact_percentage:.1f}%")
        
        return artifact_mask
    
    def extract_spectral_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract frequency domain features from EEG signals.
        
        Computes power spectral density and extracts features from different
        frequency bands that are relevant for brain state classification.
        These features capture the oscillatory patterns characteristic of
        different cognitive states.
        
        Parameters
        ----------
        data : numpy.ndarray
            EEG data with shape (samples, channels)
            
        Returns
        -------
        features : dict
            Dictionary containing spectral features for each frequency band
        """
        if data.size == 0:
            return {}
        
        # Ensure correct data format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        samples, channels = data.shape
        features = {}
        
        try:
            # Compute power spectral density using Welch's method
            # This provides better frequency resolution and noise reduction
            window_length = min(int(self.sampling_rate * 2), samples)  # 2 second windows
            overlap = window_length // 2
            
            freqs, psd = signal.welch(data, fs=self.sampling_rate, 
                                    window='hann', nperseg=window_length,
                                    noverlap=overlap, axis=0)
            
            # Extract power in each frequency band
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Find frequency indices for this band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.any(band_mask):
                    # Calculate band power (integrate PSD over frequency range)
                    band_power = np.trapz(psd[band_mask], freqs[band_mask], axis=0)
                    features[f'{band_name}_power'] = band_power
                    
                    # Calculate relative power (normalized by total power)
                    total_power = np.trapz(psd, freqs, axis=0)
                    total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero
                    features[f'{band_name}_relative'] = band_power / total_power
                    
                    # Peak frequency in this band
                    peak_indices = np.argmax(psd[band_mask], axis=0)
                    peak_freqs = freqs[band_mask][peak_indices]
                    features[f'{band_name}_peak_freq'] = peak_freqs
            
            # Additional spectral features
            # Spectral centroid (weighted mean frequency)
            spectral_centroid = np.sum(freqs[:, np.newaxis] * psd, axis=0) / np.sum(psd, axis=0)
            features['spectral_centroid'] = spectral_centroid
            
            # Spectral entropy (measure of frequency distribution)
            psd_normalized = psd / (np.sum(psd, axis=0) + 1e-10)
            spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10), axis=0)
            features['spectral_entropy'] = spectral_entropy
            
            # Alpha/beta ratio (commonly used in BCI applications)
            if 'alpha_power' in features and 'beta_power' in features:
                alpha_beta_ratio = features['alpha_power'] / (features['beta_power'] + 1e-10)
                features['alpha_beta_ratio'] = alpha_beta_ratio
            
        except Exception as e:
            self.logger.error(f"Spectral feature extraction failed: {e}")
            features = {}
        
        return features
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract time domain features from EEG signals.
        
        Computes statistical and morphological features that characterize
        the temporal structure of EEG signals. These features complement
        spectral analysis and can capture non-oscillatory neural activity.
        
        Parameters
        ----------
        data : numpy.ndarray
            EEG data with shape (samples, channels)
            
        Returns
        -------
        features : dict
            Dictionary containing temporal domain features
        """
        if data.size == 0:
            return {}
        
        # Ensure correct data format
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        features = {}
        
        try:
            # Basic statistical features
            features['mean'] = np.mean(data, axis=0)
            features['std'] = np.std(data, axis=0)
            features['variance'] = np.var(data, axis=0)
            features['skewness'] = stats.skew(data, axis=0)
            features['kurtosis'] = stats.kurtosis(data, axis=0)
            
            # Signal power and energy
            features['rms'] = np.sqrt(np.mean(data**2, axis=0))
            features['energy'] = np.sum(data**2, axis=0)
            
            # Zero crossing rate (measure of signal oscillation)
            zero_crossings = np.sum(np.diff(np.signbit(data), axis=0), axis=0)
            features['zero_crossing_rate'] = zero_crossings / data.shape[0]
            
            # Signal complexity measures
            # Hjorth parameters (activity, mobility, complexity)
            if data.shape[0] > 2:
                # Activity (variance)
                activity = np.var(data, axis=0)
                
                # Mobility (normalized variance of first derivative)
                first_diff = np.diff(data, axis=0)
                mobility = np.sqrt(np.var(first_diff, axis=0) / (activity + 1e-10))
                
                # Complexity (normalized variance of second derivative)
                if data.shape[0] > 3:
                    second_diff = np.diff(first_diff, axis=0)
                    complexity = np.sqrt(np.var(second_diff, axis=0) / (np.var(first_diff, axis=0) + 1e-10)) / (mobility + 1e-10)
                    features['hjorth_complexity'] = complexity
                
                features['hjorth_activity'] = activity
                features['hjorth_mobility'] = mobility
            
            # Peak detection and morphology
            # Count number of peaks (local maxima)
            peak_counts = np.zeros(data.shape[1])
            for ch in range(data.shape[1]):
                peaks, _ = signal.find_peaks(data[:, ch], height=np.std(data[:, ch]))
                peak_counts[ch] = len(peaks)
            features['peak_count'] = peak_counts
            
            # Signal envelope (using Hilbert transform)
            try:
                analytic_signal = signal.hilbert(data, axis=0)
                envelope = np.abs(analytic_signal)
                features['envelope_mean'] = np.mean(envelope, axis=0)
                features['envelope_std'] = np.std(envelope, axis=0)
            except:
                # Fallback if Hilbert transform fails
                pass
            
        except Exception as e:
            self.logger.error(f"Temporal feature extraction failed: {e}")
            features = {}
        
        return features
    
    def process_segment(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a segment of EEG data and extract all features.
        
        This is the main processing function that combines filtering,
        artifact detection, and feature extraction into a single pipeline.
        It's optimized for real-time operation while maintaining robustness.
        
        Parameters
        ----------
        data : numpy.ndarray
            Raw EEG data segment
            
        Returns
        -------
        features : dict
            Complete feature set extracted from the data segment
        """
        if data.size == 0:
            return {}
        
        try:
            # Step 1: Apply digital filters
            filtered_data = self.apply_filters(data)
            
            # Step 2: Detect and handle artifacts
            artifact_mask = self.detect_artifacts(filtered_data)
            
            # Step 3: Extract clean data segments
            # For now, we'll use all data but flag artifact percentage
            clean_percentage = (1 - np.mean(artifact_mask)) * 100
            
            # Step 4: Extract features from clean data
            spectral_features = self.extract_spectral_features(filtered_data)
            temporal_features = self.extract_temporal_features(filtered_data)
            
            # Combine all features
            features = {**spectral_features, **temporal_features}
            
            # Add metadata
            features['clean_percentage'] = np.array([clean_percentage])
            features['segment_length'] = np.array([data.shape[0]])
            
            self.logger.debug(f"Processed segment: {len(features)} features, "
                            f"{clean_percentage:.1f}% clean data")
            
        except Exception as e:
            self.logger.error(f"Segment processing failed: {e}")
            features = {}
        
        return features


class EEGClassifier:
    """
    Machine learning classifier for EEG-based brain state recognition.
    
    This class implements a complete machine learning pipeline for classifying
    brain states from EEG features. It supports training on historical data
    and real-time classification of incoming signals.
    
    The classifier is designed to distinguish between different mental states
    such as focus, relaxation, motor imagery, and idle states that are
    relevant for brain-computer interface applications.
    
    Parameters
    ----------
    model_type : str
        Type of machine learning model ('rf' for Random Forest, 'svm' for SVM)
    feature_scaler : bool
        Whether to apply feature scaling/normalization
        
    Attributes
    ----------
    model : sklearn estimator
        Trained machine learning model
    scaler : sklearn.preprocessing.StandardScaler
        Feature scaling transformer
    class_names : list
        Names of the classification classes
    """
    
    def __init__(self, model_type: str = 'rf', feature_scaler: bool = True):
        """Initialize the EEG classifier with specified configuration."""
        self.model_type = model_type
        self.feature_scaler = feature_scaler
        
        # Initialize model
        self.model = self._create_model()
        self.scaler = StandardScaler() if feature_scaler else None
        self.pipeline = None
        
        # Classification parameters
        self.class_names = ['rest', 'focus', 'relax', 'movement']  # Default classes
        self.is_trained = False
        
        # Feature tracking
        self.feature_names = []
        self.feature_importance = None
        
        # Performance tracking
        self.training_history = []
        self.prediction_history = deque(maxlen=100)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"EEG classifier initialized: {model_type} model, "
                        f"{'with' if feature_scaler else 'without'} scaling")
    
    def _create_model(self):
        """
        Create and configure the machine learning model.
        
        Returns
        -------
        model : sklearn estimator
            Configured machine learning model
        """
        if self.model_type == 'rf':
            # Random Forest - good for interpretability and robustness
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
        elif self.model_type == 'svm':
            # Support Vector Machine - good for high-dimensional data
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability estimates
                random_state=42
            )
        else:
            self.logger.warning(f"Unknown model type {self.model_type}, using Random Forest")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        return model
    
    def prepare_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert feature dictionary to model input format.
        
        Transforms the dictionary of extracted features into a format
        suitable for machine learning models. Handles missing features
        and ensures consistent feature ordering.
        
        Parameters
        ----------
        feature_dict : dict
            Dictionary of features from signal processing
            
        Returns
        -------
        feature_vector : numpy.ndarray
            Flattened feature vector ready for classification
        """
        if not feature_dict:
            return np.array([])
        
        # If this is the first time, establish feature names
        if not self.feature_names:
            self.feature_names = []
            for key, value in feature_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        # Multi-channel feature
                        for ch in range(len(value)):
                            self.feature_names.append(f"{key}_ch{ch}")
                    else:
                        # Single feature
                        self.feature_names.append(key)
        
        # Build feature vector
        feature_list = []
        for key, value in feature_dict.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1 and len(value) > 1:
                    # Multi-channel feature
                    feature_list.extend(value.flatten())
                else:
                    # Single feature
                    feature_list.append(float(value.flatten()[0]) if value.size > 0 else 0.0)
        
        return np.array(feature_list, dtype=np.float32)
    
    def train(self, training_data: List[Tuple[Dict, str]], validation_split: float = 0.2) -> Dict:
        """
        Train the EEG classifier on labeled data.
        
        Trains the machine learning model using a dataset of EEG features
        and corresponding class labels. Includes validation and performance
        evaluation to ensure robust classification.
        
        Parameters
        ----------
        training_data : list of tuples
            List of (feature_dict, label) pairs for training
        validation_split : float
            Fraction of data to use for validation
            
        Returns
        -------
        results : dict
            Training results including accuracy and performance metrics
        """
        if not training_data:
            self.logger.error("No training data provided")
            return {}
        
        try:
            # Prepare features and labels
            features = []
            labels = []
            
            for feature_dict, label in training_data:
                feature_vector = self.prepare_features(feature_dict)
                if feature_vector.size > 0:
                    features.append(feature_vector)
                    labels.append(label)
            
            if not features:
                self.logger.error("No valid features extracted from training data")
                return {}
            
            # Convert to arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Update class names
            self.class_names = sorted(list(set(labels)))
            
            # Split into training and validation sets
            if validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42, stratify=y
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
            
            # Create and train pipeline
            pipeline_steps = []
            if self.scaler:
                pipeline_steps.append(('scaler', self.scaler))
            pipeline_steps.append(('classifier', self.model))
            
            self.pipeline = Pipeline(pipeline_steps)
            
            # Train the model
            start_time = time.time()
            self.pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate performance
            train_score = self.pipeline.score(X_train, y_train)
            results = {
                'training_accuracy': train_score,
                'training_time': training_time,
                'num_samples': len(X_train),
                'num_features': X.shape[1],
                'class_names': self.class_names,
            }
            
            if X_val is not None:
                val_score = self.pipeline.score(X_val, y_val)
                y_pred = self.pipeline.predict(X_val)
                
                results['validation_accuracy'] = val_score
                results['classification_report'] = classification_report(y_val, y_pred, 
                                                                       target_names=self.class_names)
            
            # Feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.pipeline.named_steps['classifier'].feature_importances_
                results['feature_importance'] = self.feature_importance
            
            # Mark as trained
            self.is_trained = True
            self.training_history.append(results)
            
            self.logger.info(f"Model trained: {train_score:.3f} accuracy, "
                           f"{training_time:.2f}s, {len(self.class_names)} classes")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def predict(self, features: Dict[str, np.ndarray]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify brain state from EEG features.
        
        Performs real-time classification of the current brain state based
        on extracted EEG features. Returns both the predicted class and
        confidence scores for all classes.
        
        Parameters
        ----------
        features : dict
            Dictionary of extracted EEG features
            
        Returns
        -------
        prediction : str
            Predicted class name
        confidence : float
            Confidence score for the prediction (0-1)
        probabilities : dict
            Probability scores for all classes
        """
        if not self.is_trained:
            self.logger.error("Model not trained. Call train() first.")
            return 'unknown', 0.0, {}
        
        try:
            # Prepare features
            feature_vector = self.prepare_features(features)
            
            if feature_vector.size == 0:
                return 'unknown', 0.0, {}
            
            # Reshape for single prediction
            X = feature_vector.reshape(1, -1)
            
            # Make prediction
            prediction = self.pipeline.predict(X)[0]
            
            # Get probability scores if available
            if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
                probabilities_array = self.pipeline.predict_proba(X)[0]
                probabilities = dict(zip(self.class_names, probabilities_array))
                confidence = probabilities[prediction]
            else:
                # Fallback for models without probability estimation
                probabilities = {cls: 1.0 if cls == prediction else 0.0 for cls in self.class_names}
                confidence = 1.0
            
            # Store prediction history
            self.prediction_history.append({
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time(),
                'probabilities': probabilities.copy()
            })
            
            self.logger.debug(f"Prediction: {prediction} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            prediction = 'error'
            confidence = 0.0
            probabilities = {}
        
        return prediction, confidence, probabilities
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path where to save the model
            
        Returns
        -------
        success : bool
            True if model was saved successfully
        """
        if not self.is_trained:
            self.logger.error("No trained model to save")
            return False
        
        try:
            model_data = {
                'pipeline': self.pipeline,
                'class_names': self.class_names,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'feature_scaler': self.feature_scaler,
                'training_history': self.training_history,
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file
            
        Returns
        -------
        success : bool
            True if model was loaded successfully
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model components
            self.pipeline = model_data['pipeline']
            self.class_names = model_data['class_names']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.training_history = model_data.get('training_history', [])
            
            # Update model reference
            self.model = self.pipeline.named_steps['classifier']
            if self.feature_scaler and 'scaler' in self.pipeline.named_steps:
                self.scaler = self.pipeline.named_steps['scaler']
            
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


def main():
    """
    Main function for standalone EEG classification.
    
    Provides a command-line interface for testing the signal processing
    and classification pipeline. Can be used for model training, testing,
    and real-time classification demonstration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG Signal Processing and Classification')
    parser.add_argument('--mode', choices=['train', 'classify', 'demo'], default='demo',
                       help='Operation mode')
    parser.add_argument('--model-path', default='eeg_model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--data-path', help='Path to training data')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration for real-time demo (seconds)')
    
    args = parser.parse_args()
    
    # Initialize components
    processor = SignalProcessor(sampling_rate=250.0, num_channels=8)
    classifier = EEGClassifier(model_type='rf')
    
    if args.mode == 'demo':
        # Real-time classification demo
        print("Starting real-time EEG classification demo...")
        
        # Initialize data streamer
        streamer = BCIDataStreamer()
        
        if not streamer.connect():
            print("Failed to connect to OpenBCI. Using simulation mode.")
        
        # Load model if available
        if os.path.exists(args.model_path):
            classifier.load_model(args.model_path)
            print(f"Loaded model: {classifier.class_names}")
        else:
            print("No trained model found. Classification results will be random.")
        
        def classification_callback(data, timestamp):
            """Process and classify incoming EEG data."""
            if streamer.samples_received % 125 == 0:  # Every 0.5 seconds at 250 Hz
                # Get recent data for processing
                recent_data, _ = streamer.get_recent_data(duration_seconds=2.0)
                
                if recent_data.size > 0:
                    # Process the data
                    features = processor.process_segment(recent_data)
                    
                    if classifier.is_trained and features:
                        # Classify brain state
                        prediction, confidence, probabilities = classifier.predict(features)
                        
                        print(f"Brain State: {prediction} ({confidence:.2f}) | "
                              f"Probabilities: {probabilities}")
                    else:
                        print(f"Features extracted: {len(features)} features")
        
        # Start streaming
        if streamer.start_streaming(callback=classification_callback):
            print(f"Streaming for {args.duration} seconds. Press Ctrl+C to stop.")
            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                streamer.disconnect()
        
    elif args.mode == 'train':
        print(f"Training mode not implemented in this demo.")
        print("To train a model, provide training data in the appropriate format.")
    
    elif args.mode == 'classify':
        print("Single classification mode not implemented in this demo.")
        print("Use demo mode for real-time classification.")


if __name__ == "__main__":
    main()
# Updated on 2022-08-15
# Updated on 2022-09-02
# Updated on 2022-09-23
# Updated on 2022-09-07
# Updated on 2022-09-28
# Updated on 2022-09-30
# Updated on 2022-09-28
# Updated on 2022-09-04
# Updated on 2022-09-21
# Updated on 2022-09-01
# Updated on 2022-09-22
# Updated on 2022-10-08
# Updated on 2022-10-07
# Updated on 2022-10-17
# Updated on 2022-10-11
# Updated on 2022-10-10
# Updated on 2022-10-21
# Updated on 2022-10-18
# Updated on 2022-10-20
# Updated on 2022-10-03
# Updated on 2022-10-07
# Updated on 2022-11-03
# Updated on 2022-11-03
# Updated on 2022-11-06
# Updated on 2022-11-13
# Updated on 2022-11-17
# Updated on 2022-11-10
# Updated on 2022-11-04
# Updated on 2022-11-26
# Updated on 2022-11-04