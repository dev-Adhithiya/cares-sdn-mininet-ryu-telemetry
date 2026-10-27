import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

SCALER_PATH = "models/lstm_scaler.pkl"

# ---------------------------------------------------------
# Load scaler
# ---------------------------------------------------------

print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")

# ---------------------------------------------------------
# Expected future input format
# ---------------------------------------------------------

print("\nExpected live/sequential dataset format:")
print("""
flow_id,timestamp,flow_duration,Protocol Type,Rate,Drate,
syn_count,ack_count,fin_count,rst_count,Tot size,binary_label
""")

print(
    "Each flow_id must have multiple observations ordered by timestamp."
)

# ---------------------------------------------------------
# Sequence creation function
# ---------------------------------------------------------

def create_sequences(df, sequence_length=10):

    X_sequences = []
    y_sequences = []

    # Group data by flow
    grouped = df.groupby("flow_id")

    for flow_id, flow_data in grouped:

        # Sort observations of that flow by time
        flow_data = flow_data.sort_values("timestamp")

        # Extract features
        X = flow_data[FEATURES].values

        # Scale using training scaler
        X_scaled = scaler.transform(X)

        # Extract labels
        y = flow_data["binary_label"].values

        # Create sliding windows
        for i in range(len(flow_data) - sequence_length + 1):

            sequence = X_scaled[
                i:i + sequence_length
            ]

            # Label of the final observation
            label = y[
                i + sequence_length - 1
            ]

            X_sequences.append(sequence)
            y_sequences.append(label)

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_sequences, dtype=np.int8)
    )


# ---------------------------------------------------------
# Demonstration with synthetic sequential data
# ---------------------------------------------------------

print("\nCreating small example dataset...")

example_data = {
    "flow_id": [
        "flow_A", "flow_A", "flow_A",
        "flow_A", "flow_A", "flow_A",
        "flow_A", "flow_A", "flow_A",
        "flow_A", "flow_A", "flow_A"
    ],

    "timestamp": list(range(12)),

    "flow_duration": np.linspace(1, 12, 12),

    "Protocol Type": [6] * 12,

    "Rate": np.linspace(10, 100, 12),

    "Drate": np.zeros(12),

    "syn_count": np.linspace(0, 2, 12),

    "ack_count": np.linspace(0, 1, 12),

    "fin_count": np.zeros(12),

    "rst_count": np.linspace(0, 5, 12),

    "Tot size": np.linspace(60, 600, 12),

    "binary_label": [
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        1, 1
    ]
}

example_df = pd.DataFrame(example_data)

print("\nExample sequential data:")
print(example_df.head())

# ---------------------------------------------------------
# Create sequences
# ---------------------------------------------------------

X_seq, y_seq = create_sequences(
    example_df,
    SEQUENCE_LENGTH
)

print("\nSequence creation completed.")

print("\nX sequence shape:")
print(X_seq.shape)

print("\ny sequence shape:")
print(y_seq.shape)

print("\nSequence labels:")
print(y_seq)

print("\nExpected LSTM input format:")
print(
    "(number_of_sequences, sequence_length, number_of_features)"
)

print("\nFor this project:")
print(
    f"(samples, {SEQUENCE_LENGTH}, {len(FEATURES)})"
)

print("\nSequence pipeline is ready.")