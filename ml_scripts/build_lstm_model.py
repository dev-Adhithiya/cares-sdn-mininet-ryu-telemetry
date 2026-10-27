import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 10
NUMBER_OF_FEATURES = 9

# ---------------------------------------------------------
# Build LSTM model
# ---------------------------------------------------------

print("Building LSTM model...")

model = Sequential([
    LSTM(
        64,
        input_shape=(
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES
        )
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# ---------------------------------------------------------
# Create synthetic sequence data
#
# This is ONLY to verify that the LSTM pipeline works.
# It is NOT our research training data.
# ---------------------------------------------------------

print("\nCreating synthetic test data...")

np.random.seed(42)

X_train = np.random.randn(
    1000,
    SEQUENCE_LENGTH,
    NUMBER_OF_FEATURES
).astype(np.float32)

y_train = np.random.randint(
    0,
    2,
    size=(1000,)
).astype(np.int8)

X_validation = np.random.randn(
    200,
    SEQUENCE_LENGTH,
    NUMBER_OF_FEATURES
).astype(np.float32)

y_validation = np.random.randint(
    0,
    2,
    size=(200,)
).astype(np.int8)

print("Training shape:")
print(X_train.shape)

print("Training label shape:")
print(y_train.shape)

print("Validation shape:")
print(X_validation.shape)

# ---------------------------------------------------------
# Early stopping
# ---------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ---------------------------------------------------------
# Train
#
# Again: synthetic data only.
# ---------------------------------------------------------

print("\nTesting LSTM training pipeline...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=5,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

MODEL_PATH = "models/lstm_model_template.keras"

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nLSTM architecture test completed successfully.")

print(
    "\nIMPORTANT: This model was trained only on synthetic "
    "data and must NOT be reported as an experimental result."
)