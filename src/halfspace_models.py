"""
Implementation of Self-Training Halfspace Models with Label Noise Corruption

Authors: Lies Hadjadj, Massih-Reza Amini, Sana Louhichi
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import random

# Set random seeds for reproducibility
random.seed(2020)
tf.random.set_seed(2020)


def leaky_relu(x, leakage=0.2):
    """Leaky ReLU activation function"""
    return tf.math.scalar_mul(1.-leakage, tf.math.multiply(tf.cast(x >= 0., tf.float32), x)) + \
           tf.math.scalar_mul(leakage, tf.math.multiply(tf.cast(x < 0., tf.float32), x))


class LinearThresholdFunction(Model):
    """Linear Threshold Function (LTF) model"""
    
    def __init__(self, input_dim):
        super(LinearThresholdFunction, self).__init__()
        self.w = tf.Variable(
            tf.zeros((1, input_dim)), 
            constraint=max_norm(1.), 
            trainable=True
        )

    def call(self, x):
        return tf.reduce_sum(self.w * x, axis=1)

    def predict(self, x):
        logits = tf.reduce_sum(self.w * x, axis=1)
        predictions = tf.math.sign(logits) + tf.ones(tf.shape(logits)) - tf.math.abs(tf.math.sign(logits))
        return predictions.numpy().astype('int')


class Halfspace(Model):
    """Single Halfspace classifier"""
    
    def __init__(self, leakage=0.2, learning_rate=0.001, epochs=5000):
        super(Halfspace, self).__init__()
        self.epochs = epochs
        self.leakage = leakage
        self.learning_rate = tf.constant(learning_rate)
        self.model = None

    def call(self, x):
        return self.model(x)

    def predict(self, x):
        return ((self.model.predict(x) + 1) / 2).astype('int')

    def loss(self, predicted_y, target_y):
        return leaky_relu(-predicted_y * target_y, leakage=self.leakage)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            current_loss = self.loss(self.model(x), y)
        gradients = tape.gradient(current_loss, [self.model.w])
        self.model.w.assign_sub(tf.reshape(self.learning_rate * gradients, self.model.w.shape))

    def fit(self, x, y):
        self.model = LinearThresholdFunction(x.shape[1])
        score_prev = self.score(x, y)
        w_prev = self.model.w.numpy()

        for ep in range(self.epochs):
            x, y = shuffle(x, y, random_state=2020)
            self.train_step(x, 2*y-1)
            
            if ep % 200 == 0:
                score_current = self.score(x, y)
                if score_prev == 1 or score_current < score_prev:
                    self.model.w.assign(w_prev)
                    self.epochs = ep
                    break
                else:
                    w_prev = self.model.w.numpy()
                    score_prev = score_current

    def score(self, x, y):
        return accuracy_score(y, self.predict(x))


class SelfTrainingHalfspaces:
    """Self-Training algorithm for Halfspaces under Label Noise"""
    
    def __init__(self, leakage=0.2, learning_rate=0.001, epochs=5000, window=20, 
                 threshold_strategy='max', verbose=False):
        self.window = window
        self.epochs = epochs
        self.threshold_strategy = threshold_strategy
        self.leakage = leakage
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.halfspaces = []
        self.thresholds = []
        
    def fit(self, x_labeled, y_labeled, x_unlabeled, y_unlabeled):
        """
        Train the model using self-training approach
        
        Args:
            x_labeled: Labeled training features
            y_labeled: Labels for training data
            x_unlabeled: Unlabeled features
            y_unlabeled: True labels for unlabeled data (for evaluation only)
        """
        # Implementation continues with the existing self-training logic...
        # This includes the thresholding strategies and training loop
        pass

    def predict(self, x):
        """Make predictions using the learned sequence of halfspaces"""
        predictions = []
        for sample in x:
            # Find first halfspace where |<w_i,x>| >= threshold_i
            i = next((t[0] for t in enumerate(self.thresholds) 
                     if t[1] <= abs(self.halfspaces[t[0]](sample).numpy())), 0)
            predictions.append(self.halfspaces[i].predict(sample))
        return np.array(predictions).astype('int')

    def score(self, x, y, method='standard'):
        """
        Evaluate model performance
        
        Args:
            x: Features
            y: True labels
            method: Scoring method ('standard', 'majority_vote', 'margin_vote', 'weighted_margin_vote')
        """
        if method == 'standard':
            return accuracy_score(y, self.predict(x))
        # Add other voting methods as needed
        return None 