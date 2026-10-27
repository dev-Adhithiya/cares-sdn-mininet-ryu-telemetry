import pandas as pd

# ---------------------------------------------------------
# Results from your actual experiments
# ---------------------------------------------------------

results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9857671022075012,
        "Precision": 0.9999576154449319,
        "Recall": 0.9854656778709681,
        "F1 Score": 0.9926587571166228,
        "False Positive Rate": 0.0017322891479302754,
        "Training Time (s)": 6.41,
        "Test Inference Time (s)": 0.61,
        "True Negatives": 27661,
        "False Positives": 48,
        "False Negatives": 16702,
        "True Positives": 1132440
    },

    {
        "Model": "XGBoost",
        "Accuracy": 0.9872345776993009,
        "Precision": 0.9999488642124428,
        "Recall": 0.9869772404106716,
        "F1 Score": 0.9934207097645286,
        "False Positive Rate": 0.0020931827204157495,
        "Training Time (s)": 1.04,
        "Test Inference Time (s)": 0.58,
        "True Negatives": 27651,
        "False Positives": 58,
        "False Negatives": 14965,
        "True Positives": 1134177
    }
]

# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(results)

# ---------------------------------------------------------
# Add percentage versions
# ---------------------------------------------------------

df["Accuracy (%)"] = df["Accuracy"] * 100
df["Precision (%)"] = df["Precision"] * 100
df["Recall (%)"] = df["Recall"] * 100
df["F1 Score (%)"] = df["F1 Score"] * 100
df["False Positive Rate (%)"] = (
    df["False Positive Rate"] * 100
)

# ---------------------------------------------------------
# Reorder columns
# ---------------------------------------------------------

columns = [
    "Model",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "False Positive Rate (%)",
    "Training Time (s)",
    "Test Inference Time (s)",
    "True Negatives",
    "False Positives",
    "False Negatives",
    "True Positives"
]

comparison_df = df[columns]

# ---------------------------------------------------------
# Round values
# ---------------------------------------------------------

comparison_df = comparison_df.round({
    "Accuracy (%)": 4,
    "Precision (%)": 4,
    "Recall (%)": 4,
    "F1 Score (%)": 4,
    "False Positive Rate (%)": 4,
    "Training Time (s)": 2,
    "Test Inference Time (s)": 2
})

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\nMODEL COMPARISON")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

output_path = "results/model_comparison.csv"

comparison_df.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)

# ---------------------------------------------------------
# Simple conclusion
# ---------------------------------------------------------

print("\nSUMMARY")

print(
    "\nXGBoost achieved the highest accuracy, recall "
    "and F1 score."
)

print(
    "Random Forest achieved the lower false-positive rate."
)

print(
    "XGBoost also had faster training and slightly faster "
    "test inference in this experiment."
)