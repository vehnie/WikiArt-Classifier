"""
vit.py - Vision Transformer (ViT-B/16) for WikiArt Artist Classification

Fine-tunes a pretrained ViT-B/16 backbone (ImageNet-21K weights) with a
custom classification head for 23-class artist prediction.

Improvements over baseline ViT training:
- AdamW optimizer with weight decay (0.01)
- Cosine learning rate decay with linear warmup
- Increased dropout (0.3) and intermediate dense layer (256 units)
- Label smoothing (0.1) to reduce overconfidence
"""

import math
import tensorflow as tf
import keras_hub


def cosine_decay_with_warmup(total_steps, warmup_steps, learning_rate):
    """Create a cosine decay schedule with linear warmup.

    Parameters
    ----------
    total_steps : int
        Total number of training steps.
    warmup_steps : int
        Number of warmup steps (linear ramp from 0 to peak LR).
    learning_rate : float
        Peak learning rate after warmup.

    Returns
    -------
    tf.keras.optimizers.schedules.LearningRateSchedule
    """

    class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_steps, warmup_steps, peak_lr):
            super().__init__()
            self.total_steps = float(total_steps)
            self.warmup_steps = float(warmup_steps)
            self.peak_lr = peak_lr

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            # Linear warmup
            warmup_lr = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
            # Cosine decay
            decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
            progress = (step - self.warmup_steps) / decay_steps
            cosine_lr = self.peak_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
            return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

        def get_config(self):
            return {
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "peak_lr": self.peak_lr,
            }

    return CosineWarmup(total_steps, warmup_steps, learning_rate)


def build_vit_model(
    num_classes,
    input_shape=(224, 224, 3),
    learning_rate=3e-5,
    dropout_rate=0.3,
    dense_units=256,
    weight_decay=0.01,
    label_smoothing=0.1,
    freeze_backbone=False,
    total_steps=None,
    warmup_fraction=0.1,
):
    """Build and compile a ViT-B/16 model with a custom classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shape : tuple
        Input image shape (H, W, C).
    learning_rate : float
        Peak learning rate for AdamW optimizer.
    dropout_rate : float
        Dropout rate in the classification head.
    dense_units : int
        Units in the intermediate dense layer (0 to skip).
    weight_decay : float
        Weight decay for AdamW.
    label_smoothing : float
        Label smoothing factor for crossentropy loss.
    freeze_backbone : bool
        If True, freeze the pretrained backbone (feature-extraction mode).
    total_steps : int or None
        Total training steps (for cosine schedule). If None, uses constant LR.
    warmup_fraction : float
        Fraction of total_steps used for linear warmup.

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

    # Classification head with intermediate dense layer
    x = tf.keras.layers.Dropout(dropout_rate)(cls_token)
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="gelu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_b16")

    # 3. Optimizer — AdamW with optional cosine warmup schedule
    if total_steps is not None and total_steps > 0:
        warmup_steps = int(total_steps * warmup_fraction)
        lr_schedule = cosine_decay_with_warmup(total_steps, warmup_steps, learning_rate)
    else:
        lr_schedule = learning_rate

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
    )

    # 4. Compile with label smoothing
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            tf.keras.metrics.F1Score(average="macro", name="f1_macro"),
        ],
    )

    return model
