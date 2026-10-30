"""Retrain RF/XGBoost using actual live Mininet telemetry captures.

This script:
1. Loads all available live captures (benign.csv + udp_attack.csv)
2. Derives the 9 contract features from raw OpenFlow counters
3. Retrains Random Forest and XGBoost with a new live-domain scaler
4. Evaluates with leave-one-capture-out cross-validation
5. Saves live-adapted models for use by cares_telemetry.py

Run: python ml_scripts/retrain_on_live_captures.py
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

CONTRACT_FEATURES = [
    "flow_duration", "Protocol Type", "Rate", "Drate",
    "syn_count", "ack_count", "fin_count", "rst_count", "Tot size"
]

# All labeled live captures (path, is_attack_scenario)
LABELED_CAPTURES = [
    ("captures/benign.csv",                    False),
    ("captures/udp_attack.csv",                True),
    ("captures/benign_baseline_run1.csv",      False),
    ("captures/benign_baseline_run2.csv",      False),
    ("captures/benign_baseline_run3.csv",      False),
    ("captures/udp_flood_attack_run1.csv",     True),
    ("captures/udp_flood_attack_run2.csv",     True),
    ("captures/udp_flood_attack_run3.csv",     True),
    ("captures/tcp_syn_flood_attack_run1.csv", True),
    ("captures/tcp_syn_flood_attack_run2.csv", True),
    ("captures/tcp_syn_flood_attack_run3.csv", True),
    ("captures/icmp_flood_attack_run1.csv",    True),
    ("captures/icmp_flood_attack_run2.csv",    True),
]


def _col(df, *names):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 contract features from raw OpenFlow counters."""
    out = pd.DataFrame(index=df.index)
    dur = (_col(df, "duration_sec") + _col(df, "duration_nsec") * 1e-9).clip(lower=0.001)
    pkt = _col(df, "packet_count").clip(lower=1)
    byt = _col(df, "byte_count")

    out["flow_duration"]  = dur
    out["Protocol Type"]  = _col(df, "protocol", "Protocol Type")
    out["Rate"]           = pkt / dur
    out["Drate"]          = byt / dur
    out["syn_count"]      = _col(df, "syn_count")
    out["ack_count"]      = _col(df, "ack_count")
    out["fin_count"]      = _col(df, "fin_count")
    out["rst_count"]      = _col(df, "rst_count")
    out["Tot size"]       = byt / pkt

    return out


def load_capture(path, is_attack):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None

    # Check if already has contract features
    has_contract = all(c in df.columns for c in CONTRACT_FEATURES)
    feat_df = df[CONTRACT_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0) \
              if has_contract else derive_features(df)

    # In benign runs, all flows are 0
    if not is_attack:
        feat_df["label"] = 0
    else:
        # Attacker flow: 10.0.0.4 -> 10.0.0.5 or explicit binary_label == 1
        src_ip = df.get("src_ip", pd.Series("", index=df.index)).astype(str).str.strip()
        dst_ip = df.get("dst_ip", pd.Series("", index=df.index)).astype(str).str.strip()
        b_lbl  = df.get("binary_label", pd.Series("", index=df.index)).astype(str).str.strip()

        is_attack_flow = (
            ((src_ip == "10.0.0.4") & (dst_ip == "10.0.0.5")) |
            b_lbl.isin(["1", "1.0"])
        )
        # If no explicit endpoint matching found, fallback to 1
        if not is_attack_flow.any():
            is_attack_flow = pd.Series(True, index=df.index)

        feat_df["label"] = is_attack_flow.astype(int)

    feat_df["source"] = Path(path).stem
    return feat_df


def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr  = fp / max(fp + tn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=round(acc*100, 2), precision=round(prec*100, 2),
                recall=round(rec*100, 2), f1=round(f1*100, 2),
                fpr=round(fpr*100, 2))


def main():
    print("=" * 65)
    print("CARES-SDN: Retrain on Live Mininet Telemetry Captures")
    print("=" * 65)

    # ── load all captures ────────────────────────────────────────────────────
    frames = []
    for path, lbl in LABELED_CAPTURES:
        df = load_capture(path, lbl)
        if df is not None:
            frames.append(df)
            print(f"  Loaded {len(df):4d} rows from {path}  (default_label={lbl})")

    if not frames:
        print("ERROR: No labeled captures found.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows: {len(all_data)}")
    print(f"  Label distribution:\n{all_data['label'].value_counts()}")

    X_all = all_data[CONTRACT_FEATURES].values
    y_all = all_data["label"].astype(int).values
    src   = all_data["source"].values

    # ── leave-one-capture-out evaluation ────────────────────────────────────
    sources = np.unique(src)
    print(f"\n  Leave-one-capture-out CV across {len(sources)} captures:")

    cv_results = []
    for test_src in sources:
        train_mask = src != test_src
        test_mask  = src == test_src

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]

        if len(np.unique(y_tr)) < 2:
            print(f"    [{test_src}]  SKIP (only one class in train fold)")
            continue

        sc = StandardScaler().fit(X_tr)
        Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1)
        rf.fit(Xtr_s, y_tr)
        rf_pred = rf.predict(Xte_s)
        m = metrics(y_te, rf_pred)

        cv_results.append({"test_capture": test_src, "model": "RF", **m})
        print(f"    [{test_src}] RF  Acc={m['accuracy']}%  Prec={m['precision']}%"
              f"  Rec={m['recall']}%  F1={m['f1']}%  FPR={m['fpr']}%")

        if XGB_OK:
            xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42, verbosity=0)
            xgb.fit(Xtr_s, y_tr)
            xgb_pred = xgb.predict(Xte_s)
            mx = metrics(y_te, xgb_pred)
            cv_results.append({"test_capture": test_src, "model": "XGB", **mx})
            print(f"    [{test_src}] XGB Acc={mx['accuracy']}%  Prec={mx['precision']}%"
                  f"  Rec={mx['recall']}%  F1={mx['f1']}%  FPR={mx['fpr']}%")

    # ── full-data retraining ──────────────────────────────────────────────────
    print(f"\n  Training final models on all {len(X_all)} rows...")

    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    rf_final = RandomForestClassifier(n_estimators=150, max_depth=12,
                                       random_state=42, n_jobs=-1)
    t0 = time.perf_counter()
    rf_final.fit(X_scaled, y_all)
    rf_time = time.perf_counter() - t0
    rf_pred_all = rf_final.predict(X_scaled)
    m_rf = metrics(y_all, rf_pred_all)
    print(f"  RF   training: {rf_time:.2f}s  train Acc={m_rf['accuracy']}%  "
          f"F1={m_rf['f1']}%  FPR={m_rf['fpr']}%")

    xgb_final = None
    if XGB_OK:
        xgb_final = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric="logloss",
                                    random_state=42, verbosity=0)
        t0 = time.perf_counter()
        xgb_final.fit(X_scaled, y_all)
        xgb_time = time.perf_counter() - t0
        xgb_pred_all = xgb_final.predict(X_scaled)
        m_xgb = metrics(y_all, xgb_pred_all)
        print(f"  XGB  training: {xgb_time:.2f}s  train Acc={m_xgb['accuracy']}%  "
              f"F1={m_xgb['f1']}%  FPR={m_xgb['fpr']}%")

    # ── save live-adapted models ──────────────────────────────────────────────
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib.dump(scaler,    "models/live_scaler.pkl")
    joblib.dump(rf_final,  "models/live_random_forest.pkl")
    print(f"\n  Saved: models/live_scaler.pkl")
    print(f"  Saved: models/live_random_forest.pkl")

    if xgb_final:
        joblib.dump(xgb_final, "models/live_xgboost.pkl")
        print(f"  Saved: models/live_xgboost.pkl")

    report = {
        "training_source": "live Mininet captures",
        "captures": [c for c, _ in LABELED_CAPTURES],
        "total_rows": int(len(X_all)),
        "features": CONTRACT_FEATURES,
        "cv_results": cv_results,
        "full_train_rf": m_rf,
    }
    if xgb_final:
        report["full_train_xgb"] = m_xgb

    with open("results/live_retrain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: results/live_retrain_report.json")

    print("\nDone. Use models/live_xgboost.pkl with models/live_scaler.pkl"
          " for live inference in cares_telemetry.py.")


if __name__ == "__main__":
    main()
