# train.py
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle

class ISLModelTrainer:
    def __init__(self, data_dir='processed_data'):
        self.data_dir = data_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(self.data_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'))

        with open(os.path.join(self.data_dir, 'label_map.json'), 'r') as f:
            self.label_map = json.load(f)

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        # Convert to categorical with label smoothing
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
        y_val_cat = to_categorical(y_val_encoded, num_classes=num_classes)
        
        # Apply label smoothing (0.1) to training labels
        label_smoothing = 0.1
        y_train_cat = y_train_cat * (1 - label_smoothing) + (label_smoothing / num_classes)

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Number of classes: {num_classes}")

        return X_train, X_val, y_train_cat, y_val_cat

    def build_model(self, input_shape, num_classes):
        """Build improved dense model for landmark classification"""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            
            # First block - wider
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Second block
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.35),

            # Third block
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Fourth block
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel Architecture:")
        model.summary()
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=150):
        """Train the model with improved callbacks"""
        print("\nStarting training...")
        self.model = self.build_model(input_shape=X_train.shape[1], num_classes=y_train.shape[1])

        callbacks = [
            ModelCheckpoint(
                'isl_model_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,  # Increased for faster training
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Training complete!")

    def evaluate(self, X_val, y_val):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        y_pred_proba = self.model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)
        accuracy = np.mean(y_pred == y_true)
        print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))

        # Find problematic classes
        cm = confusion_matrix(y_true, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class Accuracy:")
        for i, (label, acc) in enumerate(zip(self.label_encoder.classes_, class_accuracy)):
            print(f"{label}: {acc*100:.1f}%")

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history saved as 'training_history.png'")
        plt.show()

    def save_label_encoder(self):
        """Save label encoder for inference"""
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("Label encoder saved as 'label_encoder.pkl'")

if __name__ == "__main__":
    trainer = ISLModelTrainer(data_dir='processed_data')
    X_train, X_val, y_train, y_val = trainer.load_data()
    trainer.train(X_train, y_train, X_val, y_val, epochs=150)
    trainer.evaluate(X_val, y_val)
    trainer.plot_history()
    trainer.save_label_encoder()
    print("\n✅ Step 2 Complete! Model trained and saved.")
