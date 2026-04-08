"""
gradcam.py — Grad-CAM heatmap generation for WikiArt models.

Supports CNN models (baseline, custom_cnn, transfer/ResNet50) and ViT.
For CNNs, computes standard Grad-CAM on the last convolutional layer.
For ViT, computes gradient-weighted activation on the patch token sequence,
reshaped to a spatial grid.

Usage:
    python src/gradcam.py --model vit --checkpoint results/models/vit.keras --output results/gradcam/
    python src/gradcam.py --model custom_cnn --checkpoint results/models/custom_cnn.keras --output results/gradcam/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import build_datasets


# ---------------------------------------------------------------------------
# Layer detection
# ---------------------------------------------------------------------------

def find_last_conv_layer(model):
    """Walk the model (and nested submodels) to find the last Conv2D layer name."""
    candidates = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            candidates.append(layer.name)
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    candidates.append(f"{layer.name}/{sub.name}")
    if not candidates:
        return None
    return candidates[-1]


def find_vit_backbone_layer(model):
    """Find the ViT backbone layer (outputs patch token sequence)."""
    for layer in model.layers:
        if "vit" in layer.name.lower() or "backbone" in layer.name.lower():
            return layer.name
    return None


# ---------------------------------------------------------------------------
# Grad-CAM for CNN models
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(model, img_array, target_class=None, last_conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a CNN model.

    Parameters
    ----------
    model : tf.keras.Model
    img_array : np.ndarray, shape (1, H, W, 3)
    target_class : int or None (uses predicted class if None)
    last_conv_layer_name : str or None (auto-detects if None)

    Returns
    -------
    heatmap : np.ndarray, shape (H_feat, W_feat), values in [0, 1]
    predicted_class : int
    confidence : float
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Handle nested layer names (e.g. "resnet50/conv5_block3_3_conv")
    if "/" in last_conv_layer_name:
        parent_name, sub_name = last_conv_layer_name.split("/", 1)
        parent_layer = model.get_layer(parent_name)
        conv_layer = parent_layer.get_layer(sub_name)
        # Build a sub-model that outputs the conv layer activations
        sub_model = tf.keras.Model(
            parent_layer.input, conv_layer.output, name="conv_extractor"
        )
        # Build the grad model
        grad_model = tf.keras.Model(
            model.input,
            [sub_model(model.get_layer(parent_name).input), model.output],
        )
    else:
        grad_model = tf.keras.Model(
            model.input,
            [model.get_layer(last_conv_layer_name).output, model.output],
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    predicted_class = int(tf.argmax(predictions[0]).numpy())
    confidence = float(predictions[0, predicted_class].numpy())

    return heatmap.numpy(), predicted_class, confidence


# ---------------------------------------------------------------------------
# Grad-CAM for ViT models
# ---------------------------------------------------------------------------

def make_gradcam_heatmap_vit(model, img_array, target_class=None, backbone_layer_name=None):
    """
    Compute gradient-weighted activation map for ViT.

    Uses the gradient of the target class w.r.t. the backbone's patch token
    outputs (excluding CLS), reshaped to a spatial grid.

    Parameters
    ----------
    model : tf.keras.Model
    img_array : np.ndarray, shape (1, H, W, 3)
    target_class : int or None
    backbone_layer_name : str or None

    Returns
    -------
    heatmap : np.ndarray, shape (grid_h, grid_w), values in [0, 1]
    predicted_class : int
    confidence : float
    """
    if backbone_layer_name is None:
        backbone_layer_name = find_vit_backbone_layer(model)

    backbone_layer = model.get_layer(backbone_layer_name)

    grad_model = tf.keras.Model(
        model.input,
        [backbone_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        backbone_output, predictions = grad_model(img_array)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, backbone_output)

    # Exclude CLS token (index 0), keep patch tokens
    patch_grads = grads[:, 1:, :]
    patch_tokens = backbone_output[:, 1:, :]

    # Weight each token by mean gradient magnitude
    weights = tf.reduce_mean(patch_grads, axis=-1)  # (1, num_patches)
    weighted = tf.reduce_sum(patch_tokens[0] * weights[0, :, tf.newaxis], axis=-1)  # (num_patches,)

    # Reshape to spatial grid (ViT-B/16 with 224x224 input -> 14x14 patches)
    num_patches = weighted.shape[0]
    grid_size = int(np.sqrt(num_patches))
    heatmap = tf.reshape(weighted, (grid_size, grid_size))
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    predicted_class = int(tf.argmax(predictions[0]).numpy())
    confidence = float(predictions[0, predicted_class].numpy())

    return heatmap.numpy(), predicted_class, confidence


# ---------------------------------------------------------------------------
# Overlay and saving utilities
# ---------------------------------------------------------------------------

def overlay_heatmap(img, heatmap, alpha=0.4, colormap="jet"):
    """Overlay a Grad-CAM heatmap on an image.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3), values in [0, 1] or [0, 255]
    heatmap : np.ndarray, shape (H_feat, W_feat), values in [0, 1]
    alpha : float, blending factor

    Returns
    -------
    superimposed : np.ndarray, shape (H, W, 3), uint8
    """
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # drop alpha channel

    # Ensure img is float [0, 1]
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0

    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)
    return superimposed


def save_gradcam(img, heatmap, save_path, true_label, pred_label, confidence,
                 artist_names=None, alpha=0.4):
    """Save a side-by-side original + Grad-CAM overlay figure."""
    superimposed = overlay_heatmap(img, heatmap, alpha=alpha)

    true_name = artist_names[true_label] if artist_names else str(true_label)
    pred_name = artist_names[pred_label] if artist_names else str(pred_label)
    correct = true_label == pred_label

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img if img.max() <= 1.0 else img.astype(np.float32) / 255.0)
    axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(superimposed)
    color = "green" if correct else "red"
    axes[1].set_title(
        f"Grad-CAM\nPred: {pred_name} ({confidence:.1%})",
        fontsize=10, color=color,
    )
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_gradcam_for_dataset(model, dataset, artist_names, output_dir,
                                 is_vit=False, max_correct=3, max_wrong=3):
    """
    Generate Grad-CAM heatmaps for correct and misclassified samples.

    Parameters
    ----------
    model : tf.keras.Model
    dataset : tf.data.Dataset yielding (images, labels) with integer labels
    artist_names : list of str
    output_dir : str
    is_vit : bool
    max_correct : int, max correct samples per class to save
    max_wrong : int, max wrong samples per class to save
    """
    os.makedirs(output_dir, exist_ok=True)

    if is_vit:
        backbone_name = find_vit_backbone_layer(model)
        gradcam_fn = lambda img, tc: make_gradcam_heatmap_vit(
            model, img, tc, backbone_name
        )
    else:
        conv_name = find_last_conv_layer(model)
        gradcam_fn = lambda img, tc: make_gradcam_heatmap(
            model, img, tc, conv_name
        )

    correct_counts = {}
    wrong_counts = {}
    total_saved = 0

    for images, labels in dataset:
        for i in range(images.shape[0]):
            img = images[i:i+1]
            true_label = int(labels[i].numpy())
            class_name = artist_names[true_label]

            heatmap, pred_label, confidence = gradcam_fn(img, None)

            correct = true_label == pred_label
            counts = correct_counts if correct else wrong_counts
            tag = "correct" if correct else "wrong"

            if counts.get(class_name, 0) >= (max_correct if correct else max_wrong):
                continue

            counts[class_name] = counts.get(class_name, 0) + 1
            idx = counts[class_name]

            save_path = os.path.join(
                output_dir,
                f"gradcam_{class_name}_{tag}_{idx}.png"
            )
            save_gradcam(
                img[0].numpy(), heatmap, save_path,
                true_label, pred_label, confidence, artist_names,
            )
            total_saved += 1

            if total_saved % 20 == 0:
                print(f"  Saved {total_saved} heatmaps...")

    print(f"Done — {total_saved} heatmaps saved to {output_dir}")
    return total_saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Heatmap Generation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (baseline, custom_cnn, transfer, vit)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .keras model")
    parser.add_argument("--output", type=str, default="results/gradcam/",
                        help="Output directory for heatmaps")
    parser.add_argument("--max_correct", type=int, default=3,
                        help="Max correct samples per class")
    parser.add_argument("--max_wrong", type=int, default=3,
                        help="Max wrong samples per class")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"\n=== Grad-CAM: {args.model} ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}\n")

    # Load test data
    import pandas as pd
    _, _, test_ds, num_classes = build_datasets(
        splits_dir="data/splits",
        img_size=(224, 224),
        batch_size=args.batch_size,
        augment_train=False,
        use_processed=True,
        processed_dir="data/processed",
    )
    test_df = pd.read_csv("data/splits/test.csv", encoding="utf-8-sig")
    artist_names = sorted(test_df["artist"].unique())

    # Load model
    model = tf.keras.models.load_model(args.checkpoint)
    print(f"Model loaded — {model.count_params():,} parameters\n")

    is_vit = args.model == "vit"

    generate_gradcam_for_dataset(
        model, test_ds, artist_names, args.output,
        is_vit=is_vit,
        max_correct=args.max_correct,
        max_wrong=args.max_wrong,
    )


if __name__ == "__main__":
    main()
