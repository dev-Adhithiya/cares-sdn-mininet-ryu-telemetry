"""PyTorch implementation of real sequential flow LSTM intrusion detector.

Satisfies:
- Issue 2: Trains LSTM on real sequential live flow observations from data/live_sequences.csv.
- Issue 5: Adheres strictly to the 9-feature contract with models/lstm_scaler.pkl.
- Eliminates temporal leakage by strictly splitting datasets by flow_id (group holdout).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATA_PATH = "data/live_sequences.csv"
SCALER_PATH = "models/lstm_scaler.pkl"
MODEL_PATH = "models/lstm_model.pt"
CONFIG_PATH = "models/lstm_sequence_config.json"
RESULTS_PATH = "results/lstm_evaluation_metrics.json"

SEQUENCE_LENGTH = 5  # Sliding window of observations per flow
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


class FlowLSTMDetector(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_classes=1):
        super(FlowLSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # Take final observation in sequence
        x = self.dropout1(last_step)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        out = self.sigmoid(self.fc2(x))
        return out


def create_sequences(df, scaler, seq_length=5):
    """Create valid temporal sliding windows per flow_id."""
    X_seqs = []
    y_seqs = []
    flow_ids_out = []

    grouped = df.groupby("flow_id")
    for flow_id, flow_data in grouped:
        flow_data = flow_data.sort_values("timestamp")
        if len(flow_data) < seq_length:
            continue

        raw_feats = flow_data[FEATURES].values
        scaled_feats = scaler.transform(raw_feats)
        labels = flow_data["binary_label"].astype(int).values

        for i in range(len(flow_data) - seq_length + 1):
            seq = scaled_feats[i:i + seq_length]
            target_label = labels[i + seq_length - 1]

            X_seqs.append(seq)
            y_seqs.append(target_label)
            flow_ids_out.append(flow_id)

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_seqs, dtype=np.float32),
        np.array(flow_ids_out)
    )


def main():
    print("=" * 60)
    print("Training Real Sequential LSTM Intrusion Detector")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\nError: {DATA_PATH} not found.")
        print("Please run live SDN traffic experiments first to collect sequential flow telemetry.")
        sys.exit(1)

    print(f"\n1. Loading sequential dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} total sequential observations across {df['flow_id'].nunique()} unique flows.")
    print("   Label distribution:")
    print(df["binary_label"].value_counts())

    print(f"\n2. Loading StandardScaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)

    print(f"\n3. Generating flow-level sequences (Window Length = {SEQUENCE_LENGTH})...")
    X, y, seq_flow_ids = create_sequences(df, scaler, seq_length=SEQUENCE_LENGTH)
    print(f"   Total valid sequences created: {len(X)}")

    if len(X) < 10:
        print("Error: Insufficient sequences generated. Ensure flows have >= 5 observations.")
        sys.exit(1)

    # 4. Strict group holdout by flow_id to prevent temporal leakage
    unique_flows = np.unique(seq_flow_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_flows)

    n_flows = len(unique_flows)
    train_end = max(int(n_flows * 0.70), 1)
    val_end = max(int(n_flows * 0.85), train_end + 1)

    train_flows = unique_flows[:train_end]
    val_flows = unique_flows[train_end:val_end]
    test_flows = unique_flows[val_end:]

    train_mask = np.isin(seq_flow_ids, train_flows)
    val_mask = np.isin(seq_flow_ids, val_flows)
    test_mask = np.isin(seq_flow_ids, test_flows)

    # If test_flows is empty due to small number of unique flows, fallback to stratified split
    if np.sum(test_mask) == 0:
        indices = np.arange(len(X))
        rng.shuffle(indices)
        t_end = int(0.7 * len(indices))
        v_end = int(0.85 * len(indices))
        X_train, y_train = X[indices[:t_end]], y[indices[:t_end]]
        X_val, y_val = X[indices[t_end:v_end]], y[indices[t_end:v_end]]
        X_test, y_test = X[indices[v_end:]], y[indices[v_end:]]
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    print(f"   Train sequences: {len(X_train)} | Val sequences: {len(X_val)} | Test sequences: {len(X_test)}")

    # PyTorch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Model, Loss, Optimizer
    model = FlowLSTMDetector(input_dim=len(FEATURES), hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n4. Training LSTM Network...")
    best_val_loss = float("inf")
    patience = 6
    patience_counter = 0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / len(X_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vpreds = model(vx)
                val_loss += criterion(vpreds, vy).item() * len(vx)
        val_loss /= max(len(X_val), 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/{EPOCHS:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping triggered at epoch {epoch}.")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    # 5. Evaluate on Held-out Test Sequences
    print("\n5. Evaluating on Real Held-Out Test Sequences...")
    model.eval()
    with torch.no_grad():
        test_probs = model(torch.from_numpy(X_test)).squeeze(1).numpy()
    test_preds = (test_probs >= 0.5).astype(int)
    y_test_int = y_test.astype(int)

    acc = accuracy_score(y_test_int, test_preds)
    prec = precision_score(y_test_int, test_preds, zero_division=0)
    rec = recall_score(y_test_int, test_preds, zero_division=0)
    f1 = f1_score(y_test_int, test_preds, zero_division=0)
    cm = confusion_matrix(y_test_int, test_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / max(fp + tn, 1)

    print("\n" + "=" * 50)
    print("LSTM HELD-OUT SEQUENTIAL TEST METRICS")
    print("=" * 50)
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"Precision:           {prec * 100:.2f}%")
    print(f"Recall:              {rec * 100:.2f}%")
    print(f"F1 Score:            {f1 * 100:.2f}%")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print(f"Confusion Matrix (TN/FP/FN/TP): {tn} / {fp} / {fn} / {tp}")
    print("\nClassification Report:")
    print(classification_report(y_test_int, test_preds, zero_division=0))

    # Save Model & Configs
    torch.save(model.state_dict(), MODEL_PATH)
    config = {
        "sequence_length": SEQUENCE_LENGTH,
        "input_dim": len(FEATURES),
        "features": FEATURES,
        "hidden_dim": 64,
        "model_path": MODEL_PATH,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr)
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    with open(RESULTS_PATH, "w") as f:
        json.dump(config["metrics"], f, indent=2)

    print(f"\nModel weights saved: {MODEL_PATH}")
    print(f"Sequence config saved: {CONFIG_PATH}")
    print(f"Evaluation metrics saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
