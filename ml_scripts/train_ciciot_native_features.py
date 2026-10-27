"""Train a reproducible binary baseline on the downloaded CICIoT2023 feature CSVs.

This uses only fields present in the downloaded files. It holds out entire attack
class folders (and benign capture files) between train, validation and test.
It is not the nine-feature/LSTM contract from the older supplied scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

FEATURES = [
    "Protocol Type", "Rate", "syn_count", "ack_count", "fin_count",
    "rst_count", "TCP", "UDP", "ICMP", "Tot size",
]
PER_FILE_ROWS = 2_000
PER_ATTACK_FOLDER_ROWS = 10_000
PER_BENIGN_FILE_ROWS = 25_000
SEED = 2026


def read_capture(path: Path, label: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=FEATURES, nrows=PER_FILE_ROWS,
                        low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    frame["binary_label"] = label
    frame["source_capture"] = f"{path.parent.name}/{path.name}"
    frame["source_group"] = path.parent.name
    return frame


def collect_split(groups: list[Path], benign_files: list[Path], split_name: str):
    frames = []
    source_count = 0
    for folder in groups:
        pieces = [read_capture(path, 1) for path in sorted(folder.glob("*.csv"))]
        pieces = [part for part in pieces if not part.empty]
        if pieces:
            group_frame = pd.concat(pieces, ignore_index=True)
            cap = min(PER_ATTACK_FOLDER_ROWS, len(group_frame))
            group_frame = group_frame.sample(n=cap, random_state=SEED)
            frames.append(group_frame)
            source_count += 1
    for path in benign_files:
        frame = read_capture(path, 0)
        if not frame.empty:
            cap = min(PER_BENIGN_FILE_ROWS, len(frame))
            frames.append(frame.sample(n=cap, random_state=SEED))
            source_count += 1
    if not frames:
        raise RuntimeError(f"No readable rows for {split_name}")
    data = pd.concat(frames, ignore_index=True)
    counts = data["binary_label"].value_counts()
    if 0 not in counts or 1 not in counts:
        raise RuntimeError(f"{split_name} is missing one binary class: {counts.to_dict()}")
    balanced_n = int(min(counts[0], counts[1]))
    data = pd.concat([
        data[data.binary_label == 0].sample(n=balanced_n, random_state=SEED),
        data[data.binary_label == 1].sample(n=balanced_n, random_state=SEED),
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return data, source_count


def metrics(name, y, pred, seconds):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Inference_seconds": seconds,
        "Evaluation_rows": int(len(y)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path(r"G:\CN\CSV\CSV"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.data_root
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")

    class_folders = sorted(p for p in root.iterdir() if p.is_dir())
    benign_folder = root / "Benign_Final"
    if benign_folder not in class_folders:
        raise SystemExit("Benign_Final folder not found")
    attack_folders = [p for p in class_folders if p.name != "Benign_Final"]
    rng = random.Random(SEED)
    rng.shuffle(attack_folders)
    n_test = math.ceil(len(attack_folders) * 0.15)
    n_validation = math.ceil(len(attack_folders) * 0.15)
    test_groups = attack_folders[:n_test]
    validation_groups = attack_folders[n_test:n_test + n_validation]
    train_groups = attack_folders[n_test + n_validation:]

    benign_files = sorted(benign_folder.glob("*.csv"))
    if len(benign_files) < 4:
        raise SystemExit("Expected at least four benign captures to separate splits")
    rng.shuffle(benign_files)
    train_benign, validation_benign, test_benign = (
        benign_files[:2], benign_files[2:3], benign_files[3:4])

    splits = {}
    split_meta = {}
    for name, groups, benign in [
        ("train", train_groups, train_benign),
        ("validation", validation_groups, validation_benign),
        ("test", test_groups, test_benign),
    ]:
        splits[name], split_meta[name] = collect_split(groups, benign, name)

    output = args.output_root
    data_dir, models_dir, results_dir = output / "data" / "ciciot_native", output / "models", output / "results"
    for directory in (data_dir, models_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    X_train, y_train = splits["train"][FEATURES], splits["train"]["binary_label"]
    X_val, y_val = splits["validation"][FEATURES], splits["validation"]["binary_label"]
    X_test, y_test = splits["test"][FEATURES], splits["test"]["binary_label"]
    result_rows = []
    models = []

    model = RandomForestClassifier(n_estimators=100, max_depth=None,
                                   random_state=SEED, n_jobs=-1,
                                   class_weight="balanced")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    val_pred = model.predict(X_val)
    val_infer = time.perf_counter() - started
    started = time.perf_counter()
    test_pred = model.predict(X_test)
    test_infer = time.perf_counter() - started
    models.append(("Random Forest", model, fit_seconds))
    result_rows.append({"Split": "validation", **metrics("Random Forest", y_val, val_pred, val_infer), "Training_seconds": fit_seconds})
    result_rows.append({"Split": "test", **metrics("Random Forest", y_test, test_pred, test_infer), "Training_seconds": fit_seconds})

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=SEED, n_jobs=-1, tree_method="hist")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        val_pred = model.predict(X_val)
        val_infer = time.perf_counter() - started
        started = time.perf_counter()
        test_pred = model.predict(X_test)
        test_infer = time.perf_counter() - started
        models.append(("XGBoost", model, fit_seconds))
        result_rows.append({"Split": "validation", **metrics("XGBoost", y_val, val_pred, val_infer), "Training_seconds": fit_seconds})
        result_rows.append({"Split": "test", **metrics("XGBoost", y_test, test_pred, test_infer), "Training_seconds": fit_seconds})
    except ImportError:
        print("xgboost is not installed; Random Forest completed, XGBoost skipped")

    import joblib
    import sklearn
    try:
        import xgboost
        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None
    for name, model, _ in models:
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"ciciot_native_{slug}.joblib")
    results = pd.DataFrame(result_rows)
    results.to_csv(results_dir / "ciciot_native_model_comparison.csv", index=False)
    report = {
        "dataset_root": str(root),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "pandas": pd.__version__, "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "xgboost": xgboost_version},
        "all_class_folders": [p.name for p in class_folders],
        "attack_folder_split": {
            "train": [p.name for p in train_groups],
            "validation": [p.name for p in validation_groups],
            "test": [p.name for p in test_groups],
        },
        "benign_capture_split": {
            "train": [p.name for p in train_benign],
            "validation": [p.name for p in validation_benign],
            "test": [p.name for p in test_benign],
        },
        "features": FEATURES,
        "sampling": {"rows_per_attack_folder_cap": PER_ATTACK_FOLDER_ROWS,
                     "rows_per_benign_capture_cap": PER_BENIGN_FILE_ROWS,
                     "rows_read_per_source_file": PER_FILE_ROWS,
                     "row_sampling_note": "Read the first 2,000 rows of each source CSV; this is a bounded exploratory sample, not a full-corpus evaluation.",
                     "balance": "downsample to minority class within each split",
                     "seed": SEED},
        "split_samples": {name: {"rows": int(len(frame)),
                                  "class_counts": {str(k): int(v) for k, v in frame.binary_label.value_counts().sort_index().items()},
                                  "source_groups": split_meta[name]}
                          for name, frame in splits.items()},
        "results": result_rows,
        "notes": [
            "The downloaded feature CSVs have no label column; binary labels were derived from class folder names.",
            "Entire attack class folders were held out by split; benign CSV capture files were held out separately.",
            "Only features actually present in the downloaded CSVs were used.",
            "This binary baseline does not reproduce the supplied nine-feature scripts or the live Ryu LSTM pipeline.",
        ],
    }
    (results_dir / "ciciot_native_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FEATURES", FEATURES)
    print("SPLIT SIZES", {n: len(f) for n, f in splits.items()})
    print(results.to_string(index=False))
    print("Saved", results_dir / "ciciot_native_model_comparison.csv")


if __name__ == "__main__":
    main()
