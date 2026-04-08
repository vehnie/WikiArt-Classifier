"""
train.py - WikiArt Training Pipeline

Usage:
    # Baseline CNN (default settings)
    python src/train.py --model baseline --epochs 30

    # Custom CNN with default hyperparameters
    python src/train.py --model custom_cnn --epochs 50

    # Custom CNN with specific hyperparameters
    python src/train.py --model custom_cnn --epochs 50 --lr 0.0003 --dropout 0.2 --filters 64
"""

import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf

# Ensure project root is on the path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import build_datasets
from models.baseline import build_baseline_model
from models.custom_cnn import build_custom_model
from models.vit import build_vit_model

def main():
    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(description='WikiArt Training Pipeline')
    parser.add_argument('--model', type=str, default='baseline', help='baseline, custom_cnn, or vit')
    parser.add_argument('--epochs', type=int, default=30, help='Max number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Custom CNN hyperparameters (defaults used when not specified)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (custom_cnn)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (custom_cnn)')
    parser.add_argument('--filters', type=int, default=32, help='Initial number of filters (custom_cnn)')
    args = parser.parse_args()

    # 2. Load datasets
    train_ds, val_ds, _, num_classes = build_datasets(
        splits_dir="data/splits",
        img_size=(224, 224),
        batch_size=args.batch_size,
        use_processed=True,
        processed_dir="data/processed"
    )

    # One-hot encode labels for categorical_crossentropy and F1Score
    train_ds_oh = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    val_ds_oh = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    # 3. Model selection
    if args.model == 'baseline':
        print("\n--- Training Baseline ---")
        model = build_baseline_model(num_classes=num_classes)

    elif args.model == 'custom_cnn':
        print(f"\n--- Training Custom CNN (lr={args.lr}, dropout={args.dropout}, filters={args.filters}) ---")
        model = build_custom_model(
            num_classes=num_classes,
            learning_rate=args.lr,
            dropout_rate=args.dropout,
            num_filters=args.filters
        )

    elif args.model == 'vit':
        print(f"\n--- Training ViT-B/16 (lr={args.lr}, dropout={args.dropout}) ---")
        model = build_vit_model(
            num_classes=num_classes,
            learning_rate=args.lr,
            dropout_rate=args.dropout,
        )
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    # 4. Compile the model (skip for custom_cnn — already compiled inside build_custom_model)
    if args.model not in ('custom_cnn', 'vit'):
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_macro')]
        )

    # 5. Callbacks
    os.makedirs('results/logs', exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    csv_logger = tf.keras.callbacks.CSVLogger(f'results/logs/{args.model}.csv')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'results/logs/{args.model}_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 6. Train the model
    print(f" Starting training for {args.model}...")
    model.fit(
        train_ds_oh,
        validation_data=val_ds_oh,
        epochs=args.epochs,
        callbacks=[early_stop, csv_logger, checkpoint]
    )

    # 7. Save the final model
    os.makedirs('results/models', exist_ok=True)
    model_path = f'results/models/{args.model}_cnn.keras'
    model.save(model_path)
    print(f" Model saved to {model_path}")

if __name__ == "__main__":
    main()