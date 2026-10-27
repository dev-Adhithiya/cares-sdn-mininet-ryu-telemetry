import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_PATH = "data/CICIOT23/train/train.csv"

RANDOM_STATE = 42

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
# Load only required columns
# ---------------------------------------------------------

print("Loading required training columns...")

df = pd.read_csv(
    TRAIN_PATH,
    usecols=FEATURES + ["label"]
)

print("Original shape:", df.shape)

# ---------------------------------------------------------
# Create binary labels
# ---------------------------------------------------------

df["binary_label"] = (
    df["label"] != "BenignTraffic"
).astype(np.int8)

print("\nOriginal binary distribution:")
print(df["binary_label"].value_counts())

# ---------------------------------------------------------
# Separate benign and malicious traffic
# ---------------------------------------------------------

benign_df = df[
    df["binary_label"] == 0
].copy()

malicious_df = df[
    df["binary_label"] == 1
].copy()

print("\nBenign samples:")
print(len(benign_df))

print("\nMalicious samples:")
print(len(malicious_df))

# ---------------------------------------------------------
# Sample malicious traffic
#
# Keep all benign traffic.
# Sample the same number of malicious rows.
# ---------------------------------------------------------

malicious_sample = malicious_df.sample(
    n=len(benign_df),
    random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# Combine both classes
# ---------------------------------------------------------

balanced_df = pd.concat(
    [
        benign_df,
        malicious_sample
    ],
    ignore_index=True
)

# Shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ---------------------------------------------------------
# Final checks
# ---------------------------------------------------------

print("\nBalanced dataset shape:")
print(balanced_df.shape)

print("\nBalanced binary distribution:")
print(
    balanced_df["binary_label"]
    .value_counts()
    .sort_index()
)

print("\nSelected features:")
for feature in FEATURES:
    print("-", feature)

print("\nChecking missing values:")
print(
    balanced_df[FEATURES]
    .isnull()
    .sum()
)

print("\nFeature statistics:")
print(
    balanced_df[FEATURES]
    .describe()
    .T
)

# ---------------------------------------------------------
# Save prepared training dataset
# ---------------------------------------------------------

OUTPUT_PATH = "data/balanced_binary_train.csv"

balanced_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nBalanced dataset saved to:")
print(OUTPUT_PATH)

print("\nTraining-data preparation completed successfully.")