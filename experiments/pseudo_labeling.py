"""
Pseudo-Labeling Experiments

This script compares different pseudo-labeling approaches:
- Linear SVM
- Single Halfspace
- Self-Training Halfspaces

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import gc
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from src.halfspace_models import Halfspace, SelfTrainingHalfspaces
from src.noise_generators import DataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run pseudo-labeling experiments')
    parser.add_argument('num_iterations', type=int,
                       help='Number of experiment iterations')
    parser.add_argument('num_features', type=int,
                       help='Number of features')
    parser.add_argument('num_samples', type=int,
                       help='Number of samples')
    parser.add_argument('--margin', type=float, default=0.05,
                       help='Margin for data generation')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results')
    return parser.parse_args()


def run_experiment(args):
    """Run the pseudo-labeling experiment."""
    # Initialize results storage
    results = {
        'err_svm': [[], []],  # [without pseudo-labels, with pseudo-labels]
        'err_halfspace': [[], []],
        'err_self_training': [[], []],
        'inferred_noise_rates': [[], [], []]  # [svm, halfspace, self-training]
    }
    
    data_generator = DataGenerator()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_iterations):
        print(f'\n*** ITERATION {i} ***')
        
        # Generate synthetic data
        X, y, true_model = data_generator.generate_margin_data(
            args.num_samples, args.num_features, args.margin
        )
        
        # Create data splits
        X_unlabeled, X_train_full, y_unlabeled, y_train_full = train_test_split(
            X, y, test_size=0.33
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.33
        )
        
        print(f'Unlabeled data size: {len(X_unlabeled)}')
        print(f'Training data size: {len(X_train)}')
        print(f'Test data size: {len(X_test)}')

        # 1. Linear SVM
        svm = SGDClassifier(max_iter=1000, tol=1e-3)
        
        # Without pseudo-labels
        svm.fit(X_train, y_train)
        results['err_svm'][0].append(1 - svm.score(X_test, y_test))
        
        # With pseudo-labels
        pseudo_labels_svm = svm.predict(X_unlabeled)
        svm.fit(X_unlabeled, pseudo_labels_svm)
        results['err_svm'][1].append(1 - svm.score(X_test, y_test))
        
        # Record inferred noise rate
        results['inferred_noise_rates'][0].append(
            np.mean(pseudo_labels_svm != y_unlabeled)
        )

        # 2. Single Halfspace
        halfspace = Halfspace()
        
        # Without pseudo-labels
        halfspace.fit(X_train, y_train)
        results['err_halfspace'][0].append(1 - halfspace.score(X_test, y_test))
        
        # With pseudo-labels
        pseudo_labels_hs = halfspace.predict(X_unlabeled)
        noise_rate = np.mean(pseudo_labels_hs != y_unlabeled)
        results['inferred_noise_rates'][1].append(noise_rate)
        
        halfspace_with_pseudo = Halfspace(leakage=noise_rate)
        X_combined = np.concatenate([X_train, X_unlabeled])
        y_combined = np.concatenate([y_train, pseudo_labels_hs])
        halfspace_with_pseudo.fit(X_combined, y_combined)
        results['err_halfspace'][1].append(
            1 - halfspace_with_pseudo.score(X_test, y_test)
        )

        # 3. Self-Training Halfspaces
        self_training = SelfTrainingHalfspaces()
        
        # Without pseudo-labels
        self_training.fit(X_train, y_train)
        results['err_self_training'][0].append(
            1 - self_training.score(X_test, y_test)
        )
        
        # With pseudo-labels
        pseudo_labels_st = self_training.predict(X_unlabeled)
        noise_rate = np.mean(pseudo_labels_st != y_unlabeled)
        results['inferred_noise_rates'][2].append(noise_rate)
        
        self_training_with_pseudo = SelfTrainingHalfspaces(leakage=noise_rate)
        self_training_with_pseudo.fit(X_combined, y_combined)
        results['err_self_training'][1].append(
            1 - self_training_with_pseudo.score(X_test, y_test)
        )

        gc.collect()

    # Save results
    for key, values in results.items():
        data_generator.save_dataset(
            values,
            f"{output_path}/ssl_{key}_{args.num_iterations}_{args.num_features}_{args.num_samples}"
        )

def main():
    args = parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main() 