"""
baseline.py - defines the baseline CNN model for WikiArt classification.

Run directly to test:
    python src/train.py --model baseline --epochs 30
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_model(num_classes, input_shape=(224, 224, 3)):
    """
    Simple CNN baseline model for WikiArt classification.
    The model consists of a few convolutional layers for feature extraction, followed by dense layers for classification.
    The architecture is intentionally kept simple to serve as a baseline for comparison with more complex models.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block: Extract basic features
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block: Deepen the network slightly
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification layers (Dense layers)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Regularization to prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model