import pandas as pd

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

train_path = "data/CICIOT23/train/train.csv"
validation_path = "data/CICIOT23/validation/validation.csv"
test_path = "data/CICIOT23/test/test.csv"

# ---------------------------------------------------------
# Function to create binary labels
# ---------------------------------------------------------

def prepare_binary_dataset(path, name):

    print(f"\nLoading {name} dataset...")

    df = pd.read_csv(path)

    print(f"{name} shape: {df.shape}")

    # BenignTraffic = 0
    # Every attack = 1
    df["binary_label"] = (
        df["label"] != "BenignTraffic"
    ).astype(int)

    print(f"\n{name} binary label distribution:")

    counts = df["binary_label"].value_counts().sort_index()

    print(counts)

    benign_count = counts.get(0, 0)
    malicious_count = counts.get(1, 0)

    print(f"\nBenign samples: {benign_count}")
    print(f"Malicious samples: {malicious_count}")

    if benign_count > 0:
        print(
            "Malicious / Benign ratio:",
            round(malicious_count / benign_count, 2)
        )

    return df


# ---------------------------------------------------------
# Prepare datasets
# ---------------------------------------------------------

train_df = prepare_binary_dataset(
    train_path,
    "TRAIN"
)

validation_df = prepare_binary_dataset(
    validation_path,
    "VALIDATION"
)

test_df = prepare_binary_dataset(
    test_path,
    "TEST"
)

# ---------------------------------------------------------
# Verify mapping
# ---------------------------------------------------------

print("\nExample label mapping:")

print(
    train_df[
        ["label", "binary_label"]
    ]
    .drop_duplicates()
    .sort_values("binary_label")
    .to_string(index=False)
)

print("\nBinary label preparation completed successfully.")