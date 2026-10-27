import os
import json
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/live_sequences.csv"

SCALER_PATH = "models/lstm_scaler.pkl"

MODEL_PATH = "models/lstm_final.keras"

SEQUENCE_CONFIG_PATH = "models/lstm_sequence_config.json"

SEQUENCE_LENGTH = 10

FEATURES = [
    "flow_duration",
    "Protocol Type",
    "Rate",
    "Drate",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "Tot size"
]

# =========================================================
# CHECK INPUT FILE
# =========================================================

if not os.path.exists(DATA_PATH):

    print("\nReal sequential dataset not found.")

    print("\nExpected file:")
    print(DATA_PATH)

    print("\nThe CSV must contain these columns:")

    print("flow_id")
    print("timestamp")

    for feature in FEATURES:
        print(feature)

    print("binary_label")

    print(
        "\nAsk the Mininet/Ryu side to generate this file "
        "before final LSTM training."
    )

    exit()

# =========================================================
# LOAD DATA
# =========================================================

print("Loading real sequential traffic data...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

# =========================================================
# VERIFY REQUIRED COLUMNS
# =========================================================

required_columns = (
    ["flow_id", "timestamp"]
    + FEATURES
    + ["binary_label"]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    exit()

print("\nAll required columns are present.")

# =========================================================
# CHECK DATA
# =========================================================

print("\nLabel distribution:")

print(
    df["binary_label"]
    .value_counts()
    .sort_index()
)

print("\nNumber of unique flows:")
print(df["flow_id"].nunique())

print("\nChecking missing values:")

print(
    df[required_columns]
    .isnull()
    .sum()
)

# =========================================================
# LOAD SCALER
# =========================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded.")

# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(dataframe):

    X_sequences = []
    y_sequences = []
    flow_ids = []

    grouped = dataframe.groupby("flow_id")

    for flow_id, flow_data in grouped:

        flow_data = flow_data.sort_values("timestamp")

        # Need enough observations for one sequence
        if len(flow_data) < SEQUENCE_LENGTH:
            continue

        # Keep DataFrame form so scaler preserves feature names
        feature_data = flow_data[FEATURES]

        scaled_features = scaler.transform(feature_data)

        labels = flow_data["binary_label"].values

        for i in range(
            len(flow_data) - SEQUENCE_LENGTH + 1
        ):

            sequence = scaled_features[
                i:i + SEQUENCE_LENGTH
            ]

            target_index = (
                i + SEQUENCE_LENGTH - 1
            )

            label = labels[target_index]

            X_sequences.append(sequence)
            y_sequences.append(label)
            flow_ids.append(flow_id)

    return (
        np.array(
            X_sequences,
            dtype=np.float32
        ),
        np.array(
            y_sequences,
            dtype=np.int8
        ),
        np.array(flow_ids)
    )


print("\nCreating LSTM sequences...")

X, y, sequence_flow_ids = create_sequences(df)

print("\nSequence shape:")
print(X.shape)

print("\nLabel shape:")
print(y.shape)

if len(X) == 0:

    print(
        "\nERROR: No sequences were created."
    )

    print(
        f"Each flow needs at least "
        f"{SEQUENCE_LENGTH} observations."
    )

    exit()

# =========================================================
# SPLIT BY FLOW
#
# IMPORTANT:
# Same flow must not appear in both training and test data.
# =========================================================

unique_flows = np.unique(sequence_flow_ids)

rng = np.random.default_rng(42)

rng.shuffle(unique_flows)

number_of_flows = len(unique_flows)

train_end = int(
    number_of_flows * 0.70
)

validation_end = int(
    number_of_flows * 0.85
)

train_flows = unique_flows[
    :train_end
]

validation_flows = unique_flows[
    train_end:validation_end
]

test_flows = unique_flows[
    validation_end:
]

train_mask = np.isin(
    sequence_flow_ids,
    train_flows
)

validation_mask = np.isin(
    sequence_flow_ids,
    validation_flows
)

test_mask = np.isin(
    sequence_flow_ids,
    test_flows
)

X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nTraining sequences:")
print(X_train.shape)

print("\nValidation sequences:")
print(X_validation.shape)

print("\nTest sequences:")
print(X_test.shape)

# =========================================================
# BUILD LSTM MODEL
# =========================================================

print("\nBuilding LSTM model...")

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            len(FEATURES)
        )
    ),

    LSTM(
        64
    ),

    Dropout(
        0.30
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")

model.summary()

# =========================================================
# EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nStarting LSTM training...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)

# =========================================================
# TEST PREDICTIONS
# =========================================================

print("\nEvaluating model...")

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# =========================================================
# FALSE POSITIVE RATE
# =========================================================

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print("\nFalse Positive Rate:")
    print(false_positive_rate)

# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)

# =========================================================
# SAVE SEQUENCE CONFIG
# =========================================================

config = {

    "sequence_length":
        SEQUENCE_LENGTH,

    "number_of_features":
        len(FEATURES),

    "features":
        FEATURES,

    "classification":
        "binary",

    "benign_label":
        0,

    "malicious_label":
        1
}

with open(
    SEQUENCE_CONFIG_PATH,
    "w"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )

print("\nSequence configuration saved to:")
print(SEQUENCE_CONFIG_PATH)

print(
    "\nReal LSTM training pipeline completed successfully."
)