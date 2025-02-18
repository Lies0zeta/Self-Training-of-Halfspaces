"""
Data Generation and Noise Models for Halfspace Learning

This module provides utilities for generating synthetic datasets and applying
different types of label noise corruption as described in the paper.

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import numpy as np
import tensorflow as tf
from scipy.stats import bernoulli, beta
import pickle
from pathlib import Path
from typing import Tuple, List, Union

from halfspace_models import LinearThresholdFunction


class DataGenerator:
    """
    Generates synthetic datasets and applies various noise models for halfspace learning experiments.
    """
    
    def __init__(self, save_dir: str = "data"):
        """
        Initialize the data generator.
        
        Args:
            save_dir: Directory to save/load generated datasets
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_margin_data(self, n_samples: int, n_features: int, 
                           margin: float) -> Tuple[np.ndarray, np.ndarray, LinearThresholdFunction]:
        """
        Generate synthetic data with specified margin from a halfspace decision boundary.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features per sample
            margin: Minimum margin from decision boundary
            
        Returns:
            Tuple containing:
            - Features matrix (n_samples x n_features)
            - Labels vector (n_samples)
            - The generating halfspace model
        """
        remaining = n_samples
        model = LinearThresholdFunction(n_features)
        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        while remaining > 0:
            # Generate batch of candidates
            batch = np.random.uniform(-1, 1, size=(n_samples, n_features))
            
            # Normalize to unit sphere
            unif = np.random.uniform(size=n_samples)
            scale_factors = np.expand_dims(
                np.linalg.norm(batch, axis=1) / unif, axis=1
            )
            batch = (batch / scale_factors).astype("float32")
            
            # Filter by margin
            margins = np.abs([model(batch[i]).numpy() for i in range(n_samples)])
            valid_batch = batch[margins >= margin]
            valid_batch = valid_batch[:min(remaining, valid_batch.shape[0])]
            
            # Get labels
            batch_labels = np.array([model.predict(valid_batch[i]) 
                                   for i in range(valid_batch.shape[0])])
            
            features.append(valid_batch)
            labels.append(batch_labels)
            remaining -= valid_batch.shape[0]

        X = np.concatenate(features)
        y = ((np.concatenate(labels) + 1) / 2).astype("int")
        return X, y, model

    def apply_noise(self, X: np.ndarray, y: np.ndarray, 
                   noise_type: str, eta: float) -> np.ndarray:
        """
        Apply specified type of label noise to the dataset.
        
        Args:
            X: Feature matrix
            y: True labels
            noise_type: Type of noise ('uniform', 'massart_random', 'massart_sigmoid', 'mixed')
            eta: Noise level parameter
            
        Returns:
            Noisy labels
        """
        noise_functions = {
            'uniform': self._uniform_noise,
            'massart_random': self._random_massart_noise,
            'massart_sigmoid': self._sigmoid_massart_noise,
            'mixed': self._mixed_noise
        }
        
        if noise_type not in noise_functions:
            raise ValueError(f"Unsupported noise type. Choose from: {list(noise_functions.keys())}")
            
        return noise_functions[noise_type](X, y, eta)

    def _sigmoid_massart_noise(self, X: np.ndarray, y: np.ndarray, 
                             eta: float) -> np.ndarray:
        """Apply sigmoid-based Massart noise."""
        y = 2 * y - 1  # Convert to {-1,1}
        alphas = np.mean(tf.math.sigmoid(X), axis=1)
        priors = np.array([beta.rvs(a, a, size=1)[0] for a in alphas]) * eta
        eta_x = 2 * bernoulli.rvs(1 - priors, size=len(y)) - 1
        y = y * eta_x
        return ((y + 1) / 2).astype("int")

    def _random_massart_noise(self, X: np.ndarray, y: np.ndarray, 
                            eta: float) -> np.ndarray:
        """Apply random Massart noise."""
        y = 2 * y - 1
        priors = np.random.uniform(0, 1, len(y)) * eta
        eta_x = 2 * bernoulli.rvs(1 - priors, size=len(y)) - 1
        y = y * eta_x
        return ((y + 1) / 2).astype("int")

    def _uniform_noise(self, X: np.ndarray, y: np.ndarray, 
                      eta: float) -> np.ndarray:
        """Apply uniform random noise."""
        y = 2 * y - 1
        eta_x = 2 * bernoulli.rvs(1 - eta, size=len(y)) - 1
        y = y * eta_x
        return ((y + 1) / 2).astype("int")

    def _mixed_noise(self, X: np.ndarray, y: np.ndarray, 
                    eta: float) -> np.ndarray:
        """Apply a mixture of different noise types."""
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        splits = np.array_split(idx, 3)
        
        noisy_labels = []
        for split_idx, noise_func in zip(splits, [
            self._uniform_noise,
            self._random_massart_noise,
            self._sigmoid_massart_noise
        ]):
            split_X = X[split_idx]
            split_y = y[split_idx]
            noisy_split = noise_func(split_X, split_y, eta)
            noisy_labels.append(noisy_split)
            
        return np.concatenate(noisy_labels)

    def save_dataset(self, data: dict, name: str) -> None:
        """Save a generated dataset."""
        with open(self.save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_dataset(self, name: str) -> dict:
        """Load a saved dataset."""
        with open(self.save_dir / f"{name}.pkl", "rb") as f:
            return pickle.load(f) 