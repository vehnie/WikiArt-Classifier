import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf
import keras_tuner as kt

# Ensure project root is on the path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import build_datasets
from models.baseline import build_baseline_model
from models.custom_cnn import custom_model_builder

def main():
    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(description='WikiArt Training Pipeline')
    parser.add_argument('--model', type=str, default='baseline', help='baseline or custom_cnn')
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
        print("\n--- Training Baseline (No Tuning) ---")
        model = build_baseline_model(num_classes=num_classes)
        
    elif args.model == 'custom_cnn':
        print("\n--- Running Random Search for Custom CNN ---")
        
        tuner = kt.RandomSearch(
            lambda hp: custom_model_builder(hp, num_classes=num_classes),
            objective='val_accuracy',
            max_trials=5,
            directory='results/tuning',    
            project_name='wikiart_custom',
            overwrite=True,
            logger=None           
        )

        tuner.search(train_ds, epochs=10, validation_data=val_ds)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
    else:
        raise ValueError(f"Model {args.model} not recognized.")
    
    # 4. Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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
        train_ds,
        validation_data=val_ds,
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