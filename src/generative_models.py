"""
Generative Models for Semi-Supervised Learning

This module implements various generative models for semi-supervised learning,
including Naive Bayes variants with self-training capabilities.

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Type
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups, load_digits


class SelfTrainingNB:
    """Self-Training with Naive Bayes classifiers"""
    
    def __init__(self, base_estimator: Type[BaseEstimator] = MultinomialNB,
                 confidence_threshold: float = 0.9,
                 max_iterations: int = 100):
        """
        Initialize Self-Training Naive Bayes model.
        
        Args:
            base_estimator: Base Naive Bayes classifier type
            confidence_threshold: Threshold for pseudo-labeling
            max_iterations: Maximum number of self-training iterations
        """
        self.base_estimator = base_estimator
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.model = None
        self.initial_model = None

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> Tuple[BaseEstimator, BaseEstimator]:
        """
        Fit the self-training model.
        
        Args:
            X_labeled: Labeled features
            y_labeled: Labels
            X_unlabeled: Unlabeled features
            
        Returns:
            Tuple of (final model, initial model)
        """
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_remain = X_unlabeled.copy()
        
        for iteration in range(self.max_iterations):
            # Train model on current labeled data
            model = self.base_estimator()
            model.fit(X_train, y_train)
            
            # Save initial model
            if iteration == 0:
                self.initial_model = model
            
            # Get predictions and confidence scores for unlabeled data
            confidences = model.predict_proba(X_remain)
            predictions = np.argmax(confidences, axis=1)
            max_confidences = confidences[np.arange(len(predictions)), predictions]
            
            # Select high confidence predictions
            selected = max_confidences >= self.confidence_threshold
            
            if not np.any(selected):
                break
                
            # Add selected samples to training data
            X_train = np.concatenate([X_train, X_remain[selected]])
            y_train = np.concatenate([y_train, predictions[selected]])
            
            # Remove selected samples from unlabeled pool
            X_remain = X_remain[~selected]
            
            if len(X_remain) == 0:
                break
                
        # Train final model
        self.model = self.base_estimator()
        self.model.fit(X_train, y_train)
        
        return self.model, self.initial_model


class DatasetLoader:
    """Loader for benchmark datasets used in the paper"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_20newsgroups(self, categories: list, target_label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess 20 Newsgroups dataset"""
        data = fetch_20newsgroups(
            subset='all',
            shuffle=False,
            remove=('headers', 'footers', 'quotes'),
            categories=categories
        )
        
        # Text preprocessing pipeline
        vectorizer = Pipeline([
            ('count', CountVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8,
                stop_words='english'
            )),
            ('tfidf', TfidfTransformer()),
        ])
        
        X = vectorizer.fit_transform(data.data)
        y = data.target
        
        # Binary classification setup
        y = (y == target_label).astype(int)
        
        return normalize(X, norm='l2', axis=0), y

    def load_digits_binary(self, positive_classes: list) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess MNIST digits for binary classification"""
        data = load_digits()
        X, y = data.data, data.target
        
        # Create binary classification
        mask = np.isin(y, positive_classes)
        X = X[mask]
        y = y[mask]
        y = (y == positive_classes[0]).astype(int)
        
        return normalize(X, norm='l2', axis=0), y

    def load_benchmark(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load benchmark datasets (Mediamill, Bibtex, etc.)"""
        data_path = self.data_dir / f"{name}.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found in {self.data_dir}")
            
        data = np.load(data_path)
        X = data['x']
        labels = data['lab']
        
        # Use most frequent class for binary classification
        class_frequencies = np.sum(labels, axis=0)
        target_class = np.argmax(class_frequencies)
        y = labels[:, target_class].astype(int)
        
        return normalize(X, norm='l2', axis=0), y 