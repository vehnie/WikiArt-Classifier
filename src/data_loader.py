"""
data_loader.py - handles loading the WikiArt dataset, splitting it, and
building tf.data pipelines for training.

Run directly to preprocess:
    python src/data_loader.py --input data/raw/ --output data/processed/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def catalog_images(raw_dir):
    """Scan raw_dir for images and build a dataframe with filepath, artist, label."""
    raw_path = Path(raw_dir)
    records = []

    for artist_dir in sorted(raw_path.iterdir()):
        if not artist_dir.is_dir():
            continue
        for f in artist_dir.iterdir():
            if f.suffix.lower() in VALID_EXTENSIONS:
                records.append({"filepath": str(f), "artist": artist_dir.name})

    df = pd.DataFrame(records)

    # encode artist names as ints (alphabetical order)
    label_map = {name: i for i, name in enumerate(sorted(df["artist"].unique()))}
    df["label"] = df["artist"].map(label_map)

    return df


def create_splits(df, splits_dir="data/splits",
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  seed=SEED):
    """Do a stratified 70/15/15 split and save the CSVs."""
    
    # first split: train vs (val+test)
    train_df, remaining = train_test_split(
        df, test_size=1 - train_ratio,
        random_state=seed, stratify=df["label"]
    )

    # second split: val vs test from the remaining chunk
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        remaining, test_size=1 - val_frac,
        random_state=seed, stratify=remaining["label"]
    )

    # save
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print(f"Splits saved -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def preprocess_and_save(df, output_dir="data/processed", img_size=IMG_SIZE):
    """Resize all images and save them to output_dir (keeps artist subdirs)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        src = Path(row["filepath"])
        dest = out_path / row["artist"] / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(img_size, Image.LANCZOS)
            img.save(dest, quality=95)
        except Exception as e:
            print(f"  skipped {src}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df)} done")

    print(f"Resized all images to {img_size} -> {output_dir}")


def build_datasets(splits_dir="data/splits", img_size=IMG_SIZE, batch_size=32,
                   augment_train=True, use_processed=False,
                   processed_dir="data/processed"):
    """
    Load split CSVs and return (train_ds, val_ds, test_ds, num_classes).
    Each dataset yields (image_batch, label_batch) with images normalised to [0,1].
    """
    import tensorflow as tf  # lazy import so the rest of the file works without tf

    splits_path = Path(splits_dir)
    train_df = pd.read_csv(splits_path / "train.csv")
    val_df   = pd.read_csv(splits_path / "val.csv")
    test_df  = pd.read_csv(splits_path / "test.csv")
    num_classes = train_df["label"].nunique()

    # optionally point filepaths to the pre-resized images
    if use_processed:
        for d in [train_df, val_df, test_df]:
            d["filepath"] = d.apply(
                lambda r: str(Path(processed_dir) / r["artist"] / Path(r["filepath"]).name),
                axis=1,
            )

    def load_img(filepath, label):
        raw = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)  # keep pixel values valid after jitter
        return img, label

    def make_dataset(dataframe, shuffle=False, do_augment=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (dataframe["filepath"].values, dataframe["label"].values)
        )
        if shuffle:
            ds = ds.shuffle(len(dataframe), seed=SEED)
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        if do_augment:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_df, shuffle=True, do_augment=augment_train)
    val_ds   = make_dataset(val_df)
    test_ds  = make_dataset(test_df)

    return train_ds, val_ds, test_ds, num_classes


# ---- CLI ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WikiArt images")
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--skip-resize", action="store_true",
                        help="only generate splits, skip resizing")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)

    print("Cataloging images ...")
    df = catalog_images(args.input)
    print(f"Found {len(df)} images across {df['artist'].nunique()} artists\n")

    print("Creating stratified splits ...")
    create_splits(df, splits_dir=args.splits)

    if not args.skip_resize:
        print("\nResizing images ...")
        preprocess_and_save(df, output_dir=args.output, img_size=img_size)

    print("\nDone.")
