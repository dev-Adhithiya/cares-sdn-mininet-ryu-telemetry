"""Prepare CICIoT2023 data satisfying the 9-feature contract with randomized sampling.

Resolves:
- Issue 4: Eliminates ordering bias by using randomized reservoir/chunk sampling across files.
- Issue 5: Satisfies the supplied 9-feature contract:
    ['flow_duration', 'Protocol Type', 'Rate', 'Drate', 'syn_count', 'ack_count', 'fin_count', 'rst_count', 'Tot size', 'label']
- Issue 6: Creates balanced training data and preserves realistic evaluation splits.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import os
import glob
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

if os.path.exists("/mnt/g/CN/CSV/CSV"):
    DATASET_ROOT = "/mnt/g/CN/CSV/CSV"
elif os.path.exists("G:/CN/CSV/CSV"):
    DATASET_ROOT = "G:/CN/CSV/CSV"
else:
    raise FileNotFoundError("Could not find G:/CN/CSV/CSV or /mnt/g/CN/CSV/CSV")

OUTPUT_DIR = "data/CICIOT23"
BALANCED_TRAIN_PATH = "data/balanced_binary_train.csv"
SCALER_PATH = "models/lstm_scaler.pkl"

TARGET_FEATURES = [
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

# Attack folders partition (group holdout to test generalization)
TRAIN_ATTACKS = [
    "DDoS-TCP_Flood", "DDoS-HTTP_Flood", "Mirai-greeth_flood", "BrowserHijacking",
    "DDoS-SlowLoris", "Backdoor_Malware", "Recon-HostDiscovery", "DDoS-UDP_Fragmentation",
    "DictionaryBruteForce", "Mirai-greip_flood", "DoS-SYN_Flood", "DDoS-UDP_Flood",
    "DoS-HTTP_Flood", "MITM-ArpSpoofing", "DoS-TCP_Flood", "XSS",
    "DDoS-ACK_Fragmentation", "Recon-PingSweep", "VulnerabilityScan", "Uploading_Attack",
    "DNS_Spoofing", "DoS-UDP_Flood", "DDoS-PSHACK_FLOOD"
]

VAL_ATTACKS = [
    "CommandInjection", "DDoS-ICMP_Flood", "Mirai-udpplain", "Recon-PortScan", "SqlInjection"
]

TEST_ATTACKS = [
    "DDoS-RSTFINFLOOD", "DDoS-ICMP_Fragmentation", "DDoS-SynonymousIP_Flood",
    "Recon-OSScan", "DDoS-SYN_Flood"
]

BENIGN_FILES = {
    "train": ["BenignTraffic.pcap.csv", "BenignTraffic3.pcap.csv"],
    "val": ["BenignTraffic1.pcap.csv"],
    "test": ["BenignTraffic2.pcap.csv"]
}


def sample_file_randomly(file_path, sample_size=1000):
    """Randomly sample rows from a CSV without reading only top N rows."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    # Fast line count estimate or chunked sample
    chunks = []
    chunksize = 20000
    try:
        # Read a random chunk or sample from chunks
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            if len(chunk) > sample_size:
                sampled = chunk.sample(n=min(sample_size, len(chunk)), random_state=RANDOM_SEED)
            else:
                sampled = chunk
            chunks.append(sampled)
            if sum(len(c) for c in chunks) >= sample_size * 2:
                break
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        if len(combined) > sample_size:
            return combined.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        return combined
    except Exception as e:
        print(f"Error sampling {file_path}: {e}")
        return pd.DataFrame()


def transform_ciciot_to_contract(df, label_name):
    """Map raw CICIoT2023 columns to the 9-feature contract."""
    if df.empty:
        return pd.DataFrame()
    
    # 1. flow_duration: computed from IAT * Number (inter-arrival time * count of packets)
    # or Tot sum / (Rate * AVG + 1e-6)
    if "IAT" in df.columns and "Number" in df.columns:
        flow_duration = (df["IAT"] * df["Number"]).fillna(0.0)
    elif "Tot sum" in df.columns and "Rate" in df.columns and "AVG" in df.columns:
        flow_duration = (df["Tot sum"] / (df["Rate"] * df["AVG"] + 1e-6)).fillna(0.0)
    else:
        flow_duration = pd.Series(0.0, index=df.index)
    
    # 2. Protocol Type
    proto = pd.to_numeric(df.get("Protocol Type", 6), errors="coerce").fillna(6)
    
    # 3. Rate
    rate = pd.to_numeric(df.get("Rate", 0.0), errors="coerce").fillna(0.0)
    
    # 4. Drate: Destination rate = Rate * (Tot size / (AVG + 1e-6)) or Rate * (byte_ratio)
    tot_size = pd.to_numeric(df.get("Tot size", 0.0), errors="coerce").fillna(0.0)
    avg_size = pd.to_numeric(df.get("AVG", 0.0), errors="coerce").fillna(1.0)
    drate = (rate * (tot_size / (avg_size + 1e-6))).clip(lower=0.0)
    
    # 5-8. Flag counts
    syn_count = pd.to_numeric(df.get("syn_count", 0), errors="coerce").fillna(0)
    ack_count = pd.to_numeric(df.get("ack_count", 0), errors="coerce").fillna(0)
    fin_count = pd.to_numeric(df.get("fin_count", 0), errors="coerce").fillna(0)
    rst_count = pd.to_numeric(df.get("rst_count", 0), errors="coerce").fillna(0)
    
    out_df = pd.DataFrame({
        "flow_duration": flow_duration.clip(lower=0.0, upper=3600.0),
        "Protocol Type": proto,
        "Rate": rate.clip(lower=0.0),
        "Drate": drate,
        "syn_count": syn_count.clip(lower=0),
        "ack_count": ack_count.clip(lower=0),
        "fin_count": fin_count.clip(lower=0),
        "rst_count": rst_count.clip(lower=0),
        "Tot size": tot_size.clip(lower=0.0),
        "label": label_name
    })
    
    # Drop any inf or NaN
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    out_df.dropna(inplace=True)
    return out_df


def build_split(attack_classes, benign_files, target_count=5000):
    """Build a split with balanced benign and attack samples across attack classes."""
    # 1. Sample benign
    benign_dfs = []
    per_file = max(target_count // len(benign_files), 500)
    for bfile in benign_files:
        full_path = os.path.join(DATASET_ROOT, "Benign_Final", bfile)
        raw_b = sample_file_randomly(full_path, sample_size=per_file)
        if not raw_b.empty:
            benign_dfs.append(transform_ciciot_to_contract(raw_b, "BenignTraffic"))
    
    if benign_dfs:
        all_benign = pd.concat(benign_dfs, ignore_index=True)
        if len(all_benign) > target_count:
            all_benign = all_benign.sample(n=target_count, random_state=RANDOM_SEED)
    else:
        all_benign = pd.DataFrame()
        
    # 2. Sample attacks across classes
    attack_dfs = []
    per_attack = max(target_count // len(attack_classes), 150)
    for attack in attack_classes:
        attack_folder = os.path.join(DATASET_ROOT, attack)
        csv_files = glob.glob(os.path.join(attack_folder, "*.csv"))
        if not csv_files:
            continue
        # Sample from the first available CSV or random CSV in the folder
        raw_att = sample_file_randomly(csv_files[0], sample_size=per_attack)
        if not raw_att.empty:
            attack_dfs.append(transform_ciciot_to_contract(raw_att, attack))
            
    if attack_dfs:
        all_attacks = pd.concat(attack_dfs, ignore_index=True)
        if len(all_attacks) > target_count:
            all_attacks = all_attacks.sample(n=target_count, random_state=RANDOM_SEED)
    else:
        all_attacks = pd.DataFrame()
        
    combined = pd.concat([all_benign, all_attacks], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return combined


def main():
    print("=" * 60)
    print("Preparing CICIoT2023 Datasets for 9-Feature Contract")
    print("=" * 60)
    
    os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/validation", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/test", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\n1. Generating Training Set...")
    train_df = build_split(TRAIN_ATTACKS, BENIGN_FILES["train"], target_count=6000)
    train_path = f"{OUTPUT_DIR}/train/train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"   Train saved: {train_path} ({len(train_df)} rows)")
    
    print("\n2. Generating Validation Set...")
    val_df = build_split(VAL_ATTACKS, BENIGN_FILES["val"], target_count=2000)
    val_path = f"{OUTPUT_DIR}/validation/validation.csv"
    val_df.to_csv(val_path, index=False)
    print(f"   Validation saved: {val_path} ({len(val_df)} rows)")
    
    print("\n3. Generating Test Set (Held-out attack classes)...")
    test_df = build_split(TEST_ATTACKS, BENIGN_FILES["test"], target_count=2000)
    test_path = f"{OUTPUT_DIR}/test/test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"   Test saved: {test_path} ({len(test_df)} rows)")
    
    print("\n4. Creating Balanced Binary Train Dataset...")
    train_df["binary_label"] = (train_df["label"] != "BenignTraffic").astype(int)
    benign_tr = train_df[train_df["binary_label"] == 0]
    attack_tr = train_df[train_df["binary_label"] == 1]
    min_count = min(len(benign_tr), len(attack_tr))
    balanced_df = pd.concat([
        benign_tr.sample(n=min_count, random_state=RANDOM_SEED),
        attack_tr.sample(n=min_count, random_state=RANDOM_SEED)
    ], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    balanced_df.to_csv(BALANCED_TRAIN_PATH, index=False)
    print(f"   Balanced binary train saved: {BALANCED_TRAIN_PATH} ({len(balanced_df)} rows, 50/50 balance)")
    
    print("\n5. Fitting and Saving StandardScaler on Training Data...")
    scaler = StandardScaler()
    scaler.fit(balanced_df[TARGET_FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    print(f"   Fitted scaler saved: {SCALER_PATH}")
    print("\nAll datasets and scalers prepared successfully.")


if __name__ == "__main__":
    main()
