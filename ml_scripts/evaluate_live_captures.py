"""Evaluate trained ML detectors against real live Ryu SDN telemetry.

Works with BOTH old-schema captures (benign.csv / udp_attack.csv) and
new-schema captures that already carry the 9-feature contract columns.

For old-schema captures the 9 features are recomputed on-the-fly from the
raw OpenFlow counters exactly as cares_telemetry.py does at runtime.

Outputs
-------
results/live_sdn_evaluation.csv
results/live_sdn_report.json

Usage (Windows or WSL):
    python  ml_scripts/evaluate_live_captures.py
    python3 ml_scripts/evaluate_live_captures.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

for _p in ["/opt/cares-venv/lib/python3.12/site-packages"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

# ── 9 contract features ──────────────────────────────────────────────────────
CONTRACT_FEATURES = [
    "flow_duration", "Protocol Type", "Rate", "Drate",
    "syn_count", "ack_count", "fin_count", "rst_count", "Tot size"
]

# ── model candidates (first found wins) ──────────────────────────────────────
MODEL_CANDIDATES = [
    "models/xgboost_model.pkl",
    "models/random_forest_model.pkl",
    "models/ciciot_native_xgboost.joblib",
    "models/ciciot_native_random_forest.joblib",
]

# ── capture files to evaluate ────────────────────────────────────────────────
CAPTURES = [
    # (path,                                      scenario,           is_attack)
    ("captures/benign.csv",                        "benign_baseline",  False),
    ("captures/udp_attack.csv",                    "udp_flood_attack", True),
    ("captures/benign_baseline_run1.csv",          "benign_run1",      False),
    ("captures/benign_baseline_run2.csv",          "benign_run2",      False),
    ("captures/benign_baseline_run3.csv",          "benign_run3",      False),
    ("captures/udp_flood_attack_run1.csv",         "udp_run1",        True),
    ("captures/udp_flood_attack_run2.csv",         "udp_run2",        True),
    ("captures/udp_flood_attack_run3.csv",         "udp_run3",        True),
    ("captures/tcp_syn_flood_attack_run1.csv",     "syn_run1",        True),
    ("captures/tcp_syn_flood_attack_run2.csv",     "syn_run2",        True),
    ("captures/tcp_syn_flood_attack_run3.csv",     "syn_run3",        True),
    ("captures/icmp_flood_attack_run1.csv",        "icmp_run1",       True),
    ("captures/icmp_flood_attack_run2.csv",        "icmp_run2",       True),
]


def _col(df: pd.DataFrame, *names):
    """Return first matching column as a numeric Series, else zeros."""
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


def derive_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9-feature contract values from raw OpenFlow counter columns."""
    out = pd.DataFrame(index=df.index)

    # flow_duration
    out["flow_duration"] = (
        _col(df, "duration_sec") + _col(df, "duration_nsec") * 1e-9
    ).clip(lower=0.001)

    # Protocol Type
    out["Protocol Type"] = _col(df, "protocol", "Protocol Type")

    # Rate / Drate
    pkt = _col(df, "packet_count").clip(lower=1)
    byt = _col(df, "byte_count")
    dur = out["flow_duration"]
    out["Rate"]  = (pkt / dur).fillna(0)
    out["Drate"] = (byt / dur).fillna(0)

    # TCP flag counts
    for col in ["syn_count", "ack_count", "fin_count", "rst_count"]:
        out[col] = _col(df, col)

    # Tot size
    out["Tot size"] = (byt / pkt).fillna(0)

    return out


def load_and_featurize(csv_path: str, is_attack: bool) -> pd.DataFrame | None:
    """Return a DataFrame with CONTRACT_FEATURES + binary_label, or None."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size < 50:
        return None

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    # ── determine whether contract columns are already present ───────────────
    has_contract = all(c in df.columns for c in CONTRACT_FEATURES)

    if has_contract:
        feat_df = df[CONTRACT_FEATURES].apply(
            pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        feat_df = derive_contract_features(df)

    # ── binary_label ─────────────────────────────────────────────────────────
    if not is_attack:
        feat_df["binary_label"] = 0
    else:
        src_ip = df.get("src_ip", pd.Series("", index=df.index)).astype(str).str.strip()
        dst_ip = df.get("dst_ip", pd.Series("", index=df.index)).astype(str).str.strip()
        b_lbl  = df.get("binary_label", pd.Series("", index=df.index)).astype(str).str.strip()

        is_attack_flow = (
            ((src_ip == "10.0.0.4") & (dst_ip == "10.0.0.5")) |
            b_lbl.isin(["1", "1.0"])
        )
        if not is_attack_flow.any():
            is_attack_flow = pd.Series(True, index=df.index)

        feat_df["binary_label"] = is_attack_flow.astype(int)

    return feat_df if len(feat_df) > 0 else None


def evaluate_with_model(feat_df: pd.DataFrame, model, scaler=None):
    """Run inference and compute binary classification metrics."""
    X = feat_df[CONTRACT_FEATURES].values
    y_true = feat_df["binary_label"].astype(int).values

    import time
    X_scaled = scaler.transform(X) if scaler is not None else X
    t0 = time.perf_counter()
    y_pred = model.predict(X_scaled).astype(int)
    mean_lat = (time.perf_counter() - t0) * 1000.0 / max(len(X), 1)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr  = fp / max(fp + tn, 1)

    return {
        "observations": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy":  round(acc  * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "f1":        round(f1   * 100, 2),
        "fpr":       round(fpr  * 100, 2),
        "mean_inference_latency_ms": round(float(mean_lat) if not np.isnan(float(mean_lat) if isinstance(mean_lat, float) else 0) else 0.0, 3),
    }


# ── model suites to evaluate ─────────────────────────────────────────────────
EVALUATION_MODELS = [
    {
        "name": "Live-Adapted XGBoost",
        "model_path": "models/live_xgboost.pkl",
        "scaler_path": "models/live_scaler.pkl",
        "domain": "live_sdn_adapted"
    },
    {
        "name": "Live-Adapted Random Forest",
        "model_path": "models/live_random_forest.pkl",
        "scaler_path": "models/live_scaler.pkl",
        "domain": "live_sdn_adapted"
    },
    {
        "name": "CICIOT23 Baseline Random Forest",
        "model_path": "models/random_forest_model.pkl",
        "scaler_path": "models/lstm_scaler.pkl",
        "domain": "ciciot23_baseline"
    },
    {
        "name": "CICIOT23 Native XGBoost",
        "model_path": "models/ciciot_native_xgboost.joblib",
        "scaler_path": "models/lstm_scaler.pkl",
        "domain": "ciciot23_native"
    }
]


def main():
    print("=" * 70)
    print("CARES-SDN Comprehensive Live Telemetry Evaluation & Benchmark")
    print("=" * 70)

    # Filter to existing models
    runnable_models = []
    for entry in EVALUATION_MODELS:
        m_path = entry["model_path"]
        if os.path.exists(m_path):
            try:
                m = joblib.load(m_path)
                s = None
                if os.path.exists(entry["scaler_path"]):
                    s = joblib.load(entry["scaler_path"])
                runnable_models.append((entry, m, s))
                print(f"Found model: {entry['name']} ({m_path})")
            except Exception as e:
                print(f"  Skip {entry['name']}: {e}")

    if not runnable_models:
        print("ERROR: No runnable models found.")
        sys.exit(1)

    all_results = []
    model_summaries = {}

    for config, model, scaler in runnable_models:
        m_name = config["name"]
        print("\n" + "-" * 70)
        print(f"EVALUATING: {m_name} (Domain: {config['domain']})")
        print("-" * 70)

        model_rows = []
        for csv_path, scenario, is_attack in CAPTURES:
            feat_df = load_and_featurize(csv_path, is_attack)
            if feat_df is None:
                continue

            try:
                metrics = evaluate_with_model(feat_df, model, scaler)
            except Exception as e:
                print(f"  [ERROR] {m_name} on {csv_path}: {e}")
                metrics = {
                    "observations": len(feat_df),
                    "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0,
                    "mean_inference_latency_ms": 0.0,
                    "error": str(e)
                }
            row = {
                "model_name": m_name,
                "model_path": config["model_path"],
                "scaler_path": config["scaler_path"],
                "domain": config["domain"],
                "scenario": scenario,
                "source_file": csv_path,
                "is_attack": is_attack,
                **metrics
            }
            model_rows.append(row)
            all_results.append(row)

            if "error" in metrics:
                continue

            flag = "ATTACK" if is_attack else "BENIGN"
            print(f"  [{flag:6s}] {scenario:25s} | Obs={metrics['observations']:4d} | "
                  f"Acc={metrics['accuracy']:6.2f}% | Prec={metrics['precision']:6.2f}% | "
                  f"Rec={metrics['recall']:6.2f}% | F1={metrics['f1']:6.2f}% | "
                  f"FPR={metrics['fpr']:5.2f}% | Lat={metrics['mean_inference_latency_ms']:.3f}ms")

        # Aggregate per model
        m_df = pd.DataFrame(model_rows)
        benign_sub = m_df[~m_df["is_attack"]]
        attack_sub = m_df[m_df["is_attack"]]

        def _calc_agg(sub):
            if sub.empty:
                return {}
            tp_sum = int(sub["tp"].sum())
            fp_sum = int(sub["fp"].sum())
            fn_sum = int(sub["fn"].sum())
            tn_sum = int(sub["tn"].sum())
            total = int(sub["observations"].sum())
            return {
                "observations": total,
                "tp": tp_sum, "fp": fp_sum, "fn": fn_sum, "tn": tn_sum,
                "accuracy": round((tp_sum + tn_sum) / max(total, 1) * 100, 2),
                "precision": round(tp_sum / max(tp_sum + fp_sum, 1) * 100, 2),
                "recall": round(tp_sum / max(tp_sum + fn_sum, 1) * 100, 2),
                "f1": round(2 * tp_sum / max(2 * tp_sum + fp_sum + fn_sum, 1) * 100, 2),
                "fpr": round(fp_sum / max(fp_sum + tn_sum, 1) * 100, 2),
                "mean_latency_ms": round(float(sub["mean_inference_latency_ms"].mean()), 3)
            }

        model_summaries[m_name] = {
            "config": config,
            "benign_metrics": _calc_agg(benign_sub),
            "attack_metrics": _calc_agg(attack_sub),
            "overall_metrics": _calc_agg(m_df)
        }

    os.makedirs("results", exist_ok=True)
    res_df = pd.DataFrame(all_results)
    res_df.to_csv("results/live_sdn_evaluation.csv", index=False)

    report = {
        "features": CONTRACT_FEATURES,
        "models_evaluated": [c["name"] for c, _, _ in runnable_models],
        "model_summaries": model_summaries,
        "detailed_results": all_results
    }
    with open("results/live_sdn_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary comparative table
    print("\n" + "=" * 80)
    print("LIVE SDN BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<30} | {'Benign FPR':<12} | {'Attack Rec':<12} | {'Overall F1':<12} | {'Latency':<10}")
    print("-" * 80)
    for m_name, s_data in model_summaries.items():
        b_fpr = s_data["benign_metrics"].get("fpr", 0.0)
        a_rec = s_data["attack_metrics"].get("recall", 0.0)
        o_f1  = s_data["overall_metrics"].get("f1", 0.0)
        lat   = s_data["overall_metrics"].get("mean_latency_ms", 0.0)
        print(f"{m_name:<30} | {b_fpr:>10.2f}% | {a_rec:>10.2f}% | {o_f1:>10.2f}% | {lat:>7.3f} ms")
    print("=" * 80)
    print(f"\nSaved CSV   : results/live_sdn_evaluation.csv")
    print(f"Saved Report: results/live_sdn_report.json")
    print("\nDone.")


if __name__ == "__main__":
    main()

