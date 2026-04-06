"""
transfer.py - Building a transfer learning model using ResNet50.

- Uses pretrained ImageNet weights
- Starts with a frozen backbone
- Can optionally fine-tune top layers
- Adds a small custom classification head

Parameters:
- num_classes: number of output classes
- input_shape: image shape
- dropout_rate: dropout in classification head
- dense_units: hidden dense layer size
- learning_rate: optimizer learning rate, configured for each experiment 
- fine_tune: whether to unfreeze top layers
- unfreeze_top_layers: number of top ResNet50 layers to unfreeze
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_transfer_model(
    num_classes,
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    dense_units=256,
    learning_rate=1e-3, #1e-5 for models 4 and 5
    fine_tune=False,
    unfreeze_top_layers=10, #5
):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base_model.trainable = False

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_top_layers]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_top_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    inputs = keras.Input(shape=input_shape)

    # Data loader outputs images in [0,1], while ResNet50 preprocess_input expects [0,255]-style input
    x = layers.Rescaling(255.0)(inputs)
    x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(x)

    x = base_model(x, training=fine_tune)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet50_transfer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


if __name__ == "__main__":
    num_classes = 23  # change this if needed
    model, base_model = build_transfer_model(num_classes=num_classes)
    print("Done")