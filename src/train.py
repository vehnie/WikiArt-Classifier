import argparse
import os
import tensorflow as tf
import pandas as pd
from data_loader import build_datasets
from models.baseline import build_baseline_model

def main():
    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(description='WikiArt Training Pipeline')
    parser.add_argument('--model', type=str, default='baseline', help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=30, help='Max number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # 2. Load datasets
    train_ds, val_ds, _, num_classes = build_datasets(
        splits_dir="data/splits",
        img_size=(224, 224),
        batch_size=args.batch_size,
        use_processed=True,
        processed_dir="data/processed"
    )

    # 3. Model selection
    if args.model == 'baseline':
        model = build_baseline_model(num_classes=num_classes)
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    # 4. Callbacks for early stopping and logging
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    os.makedirs('results/logs', exist_ok=True)
    csv_logger = tf.keras.callbacks.CSVLogger(f'results/logs/{args.model}.csv')

    # 5. Train the model
    print(f" Starting training for {args.model}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stop, csv_logger]
    )

    # 6. Save the trained model
    os.makedirs('results/models', exist_ok=True)
    model_path = f'results/models/{args.model}_cnn.keras'
    model.save(model_path)
    print(f" Model saved to {model_path}")

if __name__ == "__main__":
    main()