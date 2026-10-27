import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_PATH = "data/balanced_binary_train.csv"

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

# ---------------------------------------------------------
# Load prepared balanced dataset
# ---------------------------------------------------------

print("Loading balanced training dataset...")

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:")
print(df.shape)

# ---------------------------------------------------------
# Extract features and labels
# ---------------------------------------------------------

X = df[FEATURES].copy()
y = df["binary_label"].astype(np.int8)

print("\nFeature matrix shape:")
print(X.shape)

print("\nLabel shape:")
print(y.shape)

print("\nLabel distribution:")
print(y.value_counts().sort_index())

# ---------------------------------------------------------
# Check values before scaling
# ---------------------------------------------------------

print("\nChecking missing values:")
print(X.isnull().sum())

print("\nChecking infinite values:")

print(
    "Positive infinity:",
    np.isposinf(X.to_numpy()).sum()
)

print(
    "Negative infinity:",
    np.isneginf(X.to_numpy()).sum()
)

# ---------------------------------------------------------
# Fit StandardScaler
#
# IMPORTANT:
# This scaler is fitted ONLY on training data.
# Validation and test data must use this same scaler.
# ---------------------------------------------------------

print("\nFitting StandardScaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling completed.")

# ---------------------------------------------------------
# Check scaled data
# ---------------------------------------------------------

print("\nScaled data shape:")
print(X_scaled.shape)

print("\nFirst 5 scaled rows:")
print(X_scaled[:5])

print("\nApproximate means after scaling:")
print(np.round(X_scaled.mean(axis=0), 4))

print("\nApproximate standard deviations after scaling:")
print(np.round(X_scaled.std(axis=0), 4))

# ---------------------------------------------------------
# Save scaler
# ---------------------------------------------------------

SCALER_PATH = "models/lstm_scaler.pkl"

joblib.dump(
    scaler,
    SCALER_PATH
)

print("\nScaler saved to:")
print(SCALER_PATH)

# ---------------------------------------------------------
# Save feature order
# ---------------------------------------------------------

FEATURE_PATH = "models/lstm_features.txt"

with open(FEATURE_PATH, "w") as file:
    for feature in FEATURES:
        file.write(feature + "\n")

print("\nFeature order saved to:")
print(FEATURE_PATH)

print("\nLSTM feature preparation completed successfully.")

print("\nIMPORTANT:")
print(
    "No LSTM sequences have been created yet because "
    "the current CSV does not contain a reliable timestamp "
    "or flow/device identifier."
)