import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from datetime import datetime

# ========================
# ⚙️ CONFIG
# ========================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOG_DIR = "logs"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)


# ========================
# 📥 LOAD DATA
# ========================
def load_data():
    logging.info("Loading processed data...")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    return X_train, X_test, y_train, y_test


# ========================
# 🧠 BUILD MODEL
# ========================
def build_model(input_dim):
    logging.info("Building model...")

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),

        # Output (binary classification by default)
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ========================
# 🎯 CALLBACKS
# ========================
def get_callbacks():
    logging.info("Setting up callbacks...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),

        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True
        ),

        callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, timestamp),
            histogram_freq=1
        )
    ]

    return cb


# ========================
# 🚀 TRAIN
# ========================
def train():
    X_train, X_test, y_train, y_test = load_data()

    model = build_model(input_dim=X_train.shape[1])

    cb = get_callbacks()

    logging.info("Starting training...")

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    logging.info("Evaluating model...")

    results = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Loss: {results[0]:.4f}")
    logging.info(f"Test Accuracy: {results[1]:.4f}")
    logging.info(f"Test AUC: {results[2]:.4f}")

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "final_model")
    model.save(final_model_path)

    logging.info(f"✅ Model saved to {final_model_path}")

    return history


# ========================
# 🏁 MAIN
# ========================
if __name__ == "__main__":
    train()
