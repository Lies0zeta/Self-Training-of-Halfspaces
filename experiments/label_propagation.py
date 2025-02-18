"""
Label Propagation Experiments

This script evaluates Label Propagation and Label Spreading methods on various datasets.
Compares performance of different semi-supervised learning approaches.

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import time
import warnings
import argparse
from pathlib import Path
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score

from src.generative_models import DatasetLoader

warnings.filterwarnings("ignore")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run label propagation experiments')
    parser.add_argument('dataset', choices=[
        'comp5', 'base_hock', 'pc_mac', 'rel_ath', 'one_two', 
        'odd_even', 'mediamill', 'bibtex', 'delicious', 
        'spam', 'banknote', 'weather'
    ], help='Dataset to use')
    parser.add_argument('model', choices=['LabelPropagation', 'LabelSpreading'],
                       help='Model to use')
    parser.add_argument('labeled_size', type=int,
                       help='Number of labeled examples')
    return parser.parse_args()

def load_split_indices(split_dir: Path, dataset: str):
    """Load train/test split indices."""
    train_split = np.load(split_dir / f'trsplit{dataset}.npz')
    test_split = np.load(split_dir / f'tstsplit{dataset}.npz')
    return train_split['arr_0'], test_split['arr_0']

def run_experiment(args):
    """Run the label propagation experiment."""
    # Initialize dataset loader
    loader = DatasetLoader()
    split_dir = Path('splits')
    
    # Load data based on dataset choice
    if args.dataset in ['comp5', 'base_hock', 'pc_mac', 'rel_ath']:
        # 20 Newsgroups datasets
        categories_map = {
            'comp5': ['comp.graphics', 'comp.os.ms-windows.misc', 
                     'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
                     'comp.windows.x'],
            'base_hock': ['rec.sport.baseball', 'rec.sport.hockey'],
            'pc_mac': ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],
            'rel_ath': ['soc.religion.christian', 'talk.religion.misc', 'alt.atheism']
        }
        X, y = loader.load_20newsgroups(categories_map[args.dataset])
        
    elif args.dataset in ['one_two', 'odd_even']:
        # MNIST digit datasets
        positive_classes_map = {
            'one_two': [1, 2],
            'odd_even': [0, 2, 4, 6, 8]
        }
        X, y = loader.load_digits_binary(positive_classes_map[args.dataset])
        
    else:
        # Benchmark datasets
        X, y = loader.load_benchmark(args.dataset)

    # Load train/test splits
    train_indices, test_indices = load_split_indices(split_dir, args.dataset)
    
    # Initialize results storage
    accuracies = []
    
    # Create model
    model_class = LabelSpreading if args.model == 'LabelSpreading' else LabelPropagation
    
    print(f"Running {args.model} on {args.dataset}")
    print(f"Number of labeled examples: {args.labeled_size}")
    
    # Run experiment
    start_time = time.time()
    
    for train_idx, test_idx in zip(train_indices, test_indices):
        # Prepare data
        train_features = X[train_idx]
        train_labels = np.concatenate([
            y[train_idx[:args.labeled_size]], 
            -np.ones(len(train_idx) - args.labeled_size)
        ])
        test_features = X[test_idx]
        test_labels = y[test_idx]
        
        # Train and evaluate
        model = model_class(kernel='knn', n_jobs=-1)
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        accuracies.append(accuracy_score(test_labels, predictions))
    
    elapsed_time = time.time() - start_time
    
    # Report results
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Mean accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")

def main():
    args = parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main() 