import numpy as np
from sklearn.datasets import load_svmlight_file


class DatasetLoader:
    """
    A class to load and preprocess datasets for self-training of halfspaces.
    
    Supported datasets from LIBSVM:
    1. DNA: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    2. Pendigits: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    
    Future support planned for:
    - MNIST
    - Vehicle
    - Vowel
    """
    def __init__(self):
        self._datasets = {
            'dna': self._load_dna,
            'pendigits': self._load_pendigits
        }

    def load(self, name):
        """
        Load a dataset by name.
        
        Args:
            name (str): Name of the dataset ('dna' or 'pendigits')
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y contains the labels
            
        Raises:
            KeyError: If dataset name is not supported
        """
        if name in self._datasets:
            return self._datasets[name]()
        else:
            raise KeyError(f"Dataset '{name}' not supported. Available datasets: {list(self._datasets.keys())}")

    def _load_dna(self):
        """Load and preprocess the DNA dataset"""
        train_data = load_svmlight_file("data/dna.scale")
        test_data = load_svmlight_file("data/dna.scale.test")
        
        X_train = train_data[0].todense()
        y_train = train_data[1]
        X_test = test_data[0].todense()
        y_test = test_data[1]
        
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        
        # Transform labels to 0..K-1
        y -= 1
        return X, y

    def _load_pendigits(self):
        """Load and preprocess the Pendigits dataset"""
        data = load_svmlight_file("data/pendigits")
        X = data[0].todense()
        y = data[1]
        return X, y


def create_semisupervised_split(X_labeled, y_labeled, X_unlabeled, y_unlabeled):
    """
    Create a semi-supervised learning split from labeled and unlabeled data.
    
    Args:
        X_labeled: Features of labeled examples
        y_labeled: Labels of labeled examples
        X_unlabeled: Features of unlabeled examples
        y_unlabeled: True labels of unlabeled examples (for evaluation)
        
    Returns:
        tuple: (X_combined, y_partial, y_true_unlabeled)
            - X_combined: Combined feature matrix
            - y_partial: Labels with -1 for unlabeled data
            - y_true_unlabeled: True labels of unlabeled data
    """
    # Mark unlabeled data with -1
    y_undefined = np.repeat(-1, np.shape(X_unlabeled)[0])
    
    # Combine labeled and unlabeled data
    y_partial = np.concatenate((y_labeled, y_undefined))
    X_combined = np.concatenate((X_labeled, X_unlabeled))
    y_true = np.concatenate((y_labeled, y_unlabeled))
    
    # Shuffle the combined dataset
    n = len(y_partial)
    shuffle_idx = np.random.choice(np.arange(n), n, replace=False)
    
    X_combined = X_combined[shuffle_idx]
    y_partial = y_partial[shuffle_idx]
    y_true = y_true[shuffle_idx]
    
    # Extract true labels of unlabeled data
    y_true_unlabeled = y_true[y_partial == -1]
    
    return X_combined, y_partial, y_true_unlabeled 