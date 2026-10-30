"""Train a real PyTorch LSTM on CICIOT23 contract-feature sequential data.

This script satisfies the LSTM deliverable by:
  1. Loading the prepared CICIOT23 9-feature contract dataset (data/CICIOT23/).
  2. Constructing temporal sliding-window sequences per label group.
  3. Training a 2-layer LSTM with dropout and early stopping.
  4. Evaluating on held-out test sequences and saving model artifacts.

Run inside WSL with:
    source /opt/cares-venv/bin/activate
    python3 ml_scripts/train_lstm_ciciot.py

Or on Windows (if torch is available):
    python ml_scripts/train_lstm_ciciot.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

for _p in ["/opt/cares-venv/lib/python3.12/site-packages"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    class nn:
        Module = object

TRAIN_CSV    = "data/CICIOT23/train/train.csv"
VAL_CSV      = "data/CICIOT23/validation/validation.csv"
TEST_CSV     = "data/CICIOT23/test/test.csv"
SCALER_PATH  = "models/lstm_scaler.pkl"
MODEL_PATH   = "models/lstm_model.pt"
CONFIG_PATH  = "models/lstm_sequence_config.json"
RESULTS_PATH = "results/lstm_evaluation_metrics.json"

FEATURES = [
    "flow_duration", "Protocol Type", "Rate", "Drate",
    "syn_count", "ack_count", "fin_count", "rst_count", "Tot size"
]

SEQ_LEN    = 5
BATCH_SIZE = 128
EPOCHS     = 40
LR         = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.3
PATIENCE   = 7
SEED       = 42

if TORCH_OK:
    torch.manual_seed(SEED)
np.random.seed(SEED)


class FlowLSTMDetector(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(hidden_dim, 32)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(32, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        x      = self.dropout(last)
        x      = self.relu(self.fc1(x))
        x      = self.dropout(x)
        return self.sigmoid(self.fc2(x))


def build_sequences(df, scaler, seq_len):
    X_list, y_list = [], []
    for _, group in df.groupby("binary_label"):
        feats  = scaler.transform(group[FEATURES].values.astype(np.float32))
        labels = group["binary_label"].values.astype(np.float32)
        for i in range(len(feats) - seq_len + 1):
            X_list.append(feats[i: i + seq_len])
            y_list.append(labels[i + seq_len - 1])
    if not X_list:
        return (np.empty((0, seq_len, len(FEATURES)), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32))


def load_split(csv_path, scaler, seq_len):
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        s = df["label"].astype(str).str.strip().str.lower()
        is_benign = s.isin(["benign", "benigntraffic", "0", "false"])
        df["binary_label"] = (~is_benign).astype(int)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return build_sequences(df, scaler, seq_len)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(bx)
    return total / max(len(loader.dataset), 1)


def eval_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for bx, by in loader:
            total += criterion(model(bx), by).item() * len(bx)
    return total / max(len(loader.dataset), 1)


def main():
    print("=" * 65)
    print("CARES-SDN: Sequential LSTM Training on CICIOT23 Contract Data")
    print("=" * 65)

    if not TORCH_OK:
        print("ERROR: PyTorch not installed. Run inside /opt/cares-venv.")
        sys.exit(1)

    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV, SCALER_PATH]:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded scaler from {SCALER_PATH}")

    print(f"\nBuilding sliding-window sequences (T={SEQ_LEN})...")
    X_tr,  y_tr  = load_split(TRAIN_CSV,  scaler, SEQ_LEN)
    X_val, y_val = load_split(VAL_CSV,    scaler, SEQ_LEN)
    X_te,  y_te  = load_split(TEST_CSV,   scaler, SEQ_LEN)

    for split, X, y in [("Train", X_tr, y_tr), ("Val", X_val, y_val), ("Test", X_te, y_te)]:
        pos = int(y.sum()); neg = len(y) - pos
        print(f"  {split:5s}: {len(X):6d} seqs  (pos={pos}, neg={neg})")

    if len(X_tr) < 16:
        print("ERROR: Too few training sequences.")
        sys.exit(1)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X),
                           torch.from_numpy(y).unsqueeze(1))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    tr_loader  = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model     = FlowLSTMDetector(len(FEATURES), HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    print(f"\nTraining {EPOCHS} epochs  (hidden={HIDDEN_DIM}, layers={NUM_LAYERS},"
          f" dropout={DROPOUT}, patience={PATIENCE})...\n")

    best_val, patience_ctr, best_weights = float("inf"), 0, None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss  = train_epoch(model, tr_loader,  criterion, optimizer)
        val_loss = eval_loss(model,  val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS}  train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best val={best_val:.4f})")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    print(f"  Training time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 55)
    print("LSTM HELD-OUT TEST METRICS")
    print("=" * 55)

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X_te)).squeeze(1).numpy()
    preds = (probs >= 0.5).astype(int)
    truth = y_te.astype(int)

    acc  = accuracy_score(truth, preds)
    prec = precision_score(truth, preds, zero_division=0)
    rec  = recall_score(truth, preds, zero_division=0)
    f1   = f1_score(truth, preds, zero_division=0)
    cm   = confusion_matrix(truth, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr  = fp / max(fp + tn, 1)

    print(f"Accuracy :  {acc *100:.2f}%")
    print(f"Precision:  {prec*100:.2f}%")
    print(f"Recall   :  {rec *100:.2f}%")
    print(f"F1       :  {f1  *100:.2f}%")
    print(f"FPR      :  {fpr *100:.2f}%")
    print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("\nClassification Report:")
    print(classification_report(truth, preds, zero_division=0))

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)

    config = {
        "training_source": "CICIOT23 9-feature contract dataset",
        "sequence_length": SEQ_LEN,
        "input_dim":       len(FEATURES),
        "features":        FEATURES,
        "hidden_dim":      HIDDEN_DIM,
        "num_layers":      NUM_LAYERS,
        "model_path":      MODEL_PATH,
        "metrics": {
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "fpr":       float(fpr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }
    with open(CONFIG_PATH,  "w") as f:
        json.dump(config, f, indent=2)
    with open(RESULTS_PATH, "w") as f:
        json.dump(config["metrics"], f, indent=2)

    print(f"\nSaved model  : {MODEL_PATH}")
    print(f"Saved config : {CONFIG_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
