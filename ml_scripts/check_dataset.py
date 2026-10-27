import pandas as pd

train_path = "data/CICIOT23/train/train.csv"

print("Loading training dataset...")

df = pd.read_csv(train_path)

print("\nDataset loaded successfully.")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nNumber of unique labels:")
print(df["label"].nunique())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nInfinite values check:")
numeric_df = df.select_dtypes(include=["number"])

print("Positive infinity values:")
print((numeric_df == float("inf")).sum())

print("\nNegative infinity values:")
print((numeric_df == float("-inf")).sum())