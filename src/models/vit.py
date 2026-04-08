"""
vit.py - Vision Transformer (ViT-B/16) for WikiArt Artist Classification

Fine-tunes a pretrained ViT-B/16 backbone (ImageNet-21K weights) with a
custom classification head for 23-class artist prediction.
"""

import tensorflow as tf
import keras_hub


def build_vit_model(
    num_classes,
    input_shape=(224, 224, 3),
    learning_rate=3e-5,
    dropout_rate=0.1,
    freeze_backbone=False,
):
    """Build and compile a ViT-B/16 model with a custom classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shape : tuple
        Input image shape (H, W, C).
    learning_rate : float
        Learning rate for Adam optimizer (default 3e-5, standard for ViT fine-tuning).
    dropout_rate : float
        Dropout rate before the classification head.
    freeze_backbone : bool
        If True, freeze the pretrained backbone (feature-extraction mode).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    # 1. Load pretrained ViT-B/16 backbone
    backbone = keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet21k")
    backbone.trainable = not freeze_backbone

    # 2. Build the model
    inputs = tf.keras.Input(shape=input_shape)

    # ImageNet normalization on [0,1]-scaled inputs (keeps data_loader.py unchanged)
    x = tf.keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
    )(inputs)

    # Backbone forward pass — output includes CLS token representation
    backbone_output = backbone(x)

    # Extract CLS token (first token in the sequence)
    cls_token = backbone_output[:, 0, :]

    x = tf.keras.layers.Dropout(dropout_rate)(cls_token)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_b16")

    # 3. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.F1Score(average="macro", name="f1_macro"),
        ],
    )

    return model
