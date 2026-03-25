"""
custom_cnn.py - defines the custom CNN model for WikiArt classification.

Run directly to test:
    python src/train.py --model custom_cnn --epochs 50
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_model(num_classes, input_shape=(224, 224, 3), 
                       learning_rate=1e-3, dropout_rate=0.5, num_filters=32):
    """
    Deeper CNN architecture with Batch Normalization and Dropout.
    Parametrized for Hyperparameter Tuning (Random Search).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First block: Start with a base number of filters, 
        # in order to allow for more complex feature extraction in deeper layers
        layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second block: Double the number of filters, to capture more complex features
        layers.Conv2D(num_filters * 2, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third block: Double the number of filters again, 
        # to further increase the model's capacity to learn complex patterns
        layers.Conv2D(num_filters * 4, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Fourth block: Last convolutional block with the highest number of filters,
        # to capture the most complex features before classification
        layers.Conv2D(num_filters * 8, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate), 
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_macro')]
    )

    return model

def custom_model_builder(hp, num_classes, input_shape=(224, 224, 3)):
    """
    Auxiliary function for Keras Tuner to build the custom model with hyperparameters.
    This function will be called by the Tuner to create models with different hyperparameter combinations.
    """
    # Different hyperparameters to search over
    hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    hp_dropout = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)
    hp_filters = hp.Choice('num_filters', values=[16, 32, 64])

    # Return the model built with the current set of hyperparameters
    return build_custom_model(
        num_classes=num_classes,
        input_shape=input_shape,
        learning_rate=hp_lr,
        dropout_rate=hp_dropout,
        num_filters=hp_filters
    )