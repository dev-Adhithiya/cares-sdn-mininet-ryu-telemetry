import pandas as pd

train_path = "data/CICIOT23/train/train.csv"

features = [
    "flow_duration",
    "Protocol Type",
    "Rate",
    "Srate",
    "Drate",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "TCP",
    "UDP",
    "ICMP",
    "Tot size"
]

print("Loading dataset...")

df = pd.read_csv(
    train_path,
    usecols=features + ["label"]
)

print("\nSelected feature set:")
for feature in features:
    print("-", feature)

print("\nShape after feature selection:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nFeature statistics:")
print(df[features].describe().T)

print("\nChecking missing values:")
print(df[features].isnull().sum())

print("\nCreating binary label...")
df["binary_label"] = (
    df["label"] != "BenignTraffic"
).astype(int)

print("\nBinary distribution:")
print(df["binary_label"].value_counts())

print("\nFeature selection check completed successfully.")