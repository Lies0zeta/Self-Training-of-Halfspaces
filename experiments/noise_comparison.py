"""
Noise Model Comparison Experiments

This script compares different classification approaches under various noise conditions:
- Random Classification Noise (RCN)
- Random Massart Noise (RMN)
- Sigmoid Massart Noise (SMN)
- Mixed Noise (MXT)

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import gc
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

from src.halfspace_models import Halfspace, SelfTrainingHalfspaces
from src.noise_generators import DataGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run noise comparison experiments')
    parser.add_argument('noise_type', choices=['RCN', 'RMN', 'SMN', 'MXT'],
                       help='Type of noise to apply')
    parser.add_argument('num_iterations', type=int,
                       help='Number of experiment iterations')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results')
    return parser.parse_args()

def get_noise_function(noise_type: str):
    """Map noise type to corresponding noise function."""
    noise_mapping = {
        'RCN': 'uniform',
        'RMN': 'massart_random',
        'SMN': 'massart_sigmoid',
        'MXT': 'mixed'
    }
    return noise_mapping[noise_type]

def run_experiment(noise_type: str, num_iterations: int, output_dir: str):
    """
    Run the noise comparison experiment.
    
    Args:
        noise_type: Type of noise to apply
        num_iterations: Number of experiment iterations
        output_dir: Directory to save results
    """
    # Initialize results storage
    results = {
        'etas': [], 'gammas': [], 'epsilons': [],
        'err_bayes': [], 'err_svm': [], 
        'err_halfspace': [], 'err_self_training': []
    }
    
    data_generator = DataGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_iterations):
        # Generate random parameters
        n_features = int(100 * np.random.uniform(0.01, 1))
        gamma = np.random.uniform(0.01, 0.1)
        eps = np.random.uniform(0.01, 0.02)
        n_samples = int(np.log(1/(gamma * eps)) * (1/(eps**2)))
        eta = 0.49999 * np.random.uniform(0, 1)
        window_size = int(np.log(1/(gamma**2 * eps**4)))
        
        print(f'\n*** ITERATION {i} ***')
        print(f'Parameters: gamma={gamma:.4f}, eps={eps:.4f}, '
              f'n_samples={n_samples}, n_features={n_features}, '
              f'eta={eta:.4f}, window_size={window_size}')

        # Generate data
        X, y, true_model = data_generator.generate_margin_data(
            n_samples, n_features, gamma
        )
        
        # Apply noise
        noisy_y = data_generator.apply_noise(X, y, get_noise_function(noise_type), eta)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, noisy_y, test_size=0.33
        )

        # Evaluate models
        # 1. Bayes classifier (true model)
        bayes_clf = Halfspace()
        bayes_clf.set_model(true_model)
        err_bayes = 1 - bayes_clf.score(X_test, y_test)

        # 2. Linear SVM
        svm = SGDClassifier()
        svm.fit(X_train, y_train)
        err_svm = 1 - svm.score(X_test, y_test)

        # 3. Single Halfspace
        halfspace = Halfspace(leakage=eta, epochs=1)
        halfspace.fit(X_train, y_train)
        err_halfspace = 1 - halfspace.score(X_test, y_test)

        # 4. Self-Training Halfspaces
        self_training = SelfTrainingHalfspaces(
            leakage=eta, epochs=1, window=window_size
        )
        self_training.fit(X_train, y_train)
        err_self_training = 1 - self_training.score(X_test, y_test)

        # Store results
        results['etas'].append(eta)
        results['gammas'].append(gamma)
        results['epsilons'].append(eps)
        results['err_bayes'].append(err_bayes)
        results['err_svm'].append(err_svm)
        results['err_halfspace'].append(err_halfspace)
        results['err_self_training'].append(err_self_training)

        gc.collect()

    # Save results
    for key, values in results.items():
        data_generator.save_dataset(
            values, 
            f"{output_path}/{noise_type}_{key}_{num_iterations}"
        )

def main():
    args = parse_args()
    run_experiment(args.noise_type, args.num_iterations, args.output_dir)

if __name__ == "__main__":
    main() 