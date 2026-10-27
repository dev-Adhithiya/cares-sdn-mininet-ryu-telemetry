import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_PATH = "data/balanced_binary_train.csv"

VALIDATION_PATH = "data/CICIOT23/validation/validation.csv"
TEST_PATH = "data/CICIOT23/test/test.csv"

MODEL_PATH = "models/random_forest_model.pkl"

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

# ---------------------------------------------------------
# Load balanced training data
# ---------------------------------------------------------

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df[FEATURES]
y_train = train_df["binary_label"]

print("Training shape:")
print(X_train.shape)

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

# ---------------------------------------------------------
# Load validation data
# ---------------------------------------------------------

print("\nLoading validation dataset...")

validation_df = pd.read_csv(
    VALIDATION_PATH,
    usecols=FEATURES + ["label"]
)

validation_df["binary_label"] = (
    validation_df["label"] != "BenignTraffic"
).astype(int)

X_validation = validation_df[FEATURES]
y_validation = validation_df["binary_label"]

print("Validation shape:")
print(X_validation.shape)

# ---------------------------------------------------------
# Build Random Forest
# ---------------------------------------------------------

print("\nBuilding Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining Random Forest...")

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("\nTraining completed.")

print("Training time:")
print(round(training_time, 2), "seconds")

# ---------------------------------------------------------
# Validation prediction
# ---------------------------------------------------------

print("\nEvaluating on validation data...")

start_time = time.time()

validation_predictions = model.predict(
    X_validation
)

validation_inference_time = time.time() - start_time

# ---------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_cm = confusion_matrix(
    y_validation,
    validation_predictions
)

print("\nVALIDATION RESULTS")

print("\nAccuracy:")
print(validation_accuracy)

print("\nPrecision:")
print(validation_precision)

print("\nRecall:")
print(validation_recall)

print("\nF1 Score:")
print(validation_f1)

print("\nConfusion Matrix:")
print(validation_cm)

if validation_cm.shape == (2, 2):

    tn, fp, fn, tp = validation_cm.ravel()

    validation_fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print("\nFalse Positive Rate:")
    print(validation_fpr)

print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0
    )
)

print("\nValidation inference time:")
print(
    round(
        validation_inference_time,
        2
    ),
    "seconds"
)

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    usecols=FEATURES + ["label"]
)

test_df["binary_label"] = (
    test_df["label"] != "BenignTraffic"
).astype(int)

X_test = test_df[FEATURES]
y_test = test_df["binary_label"]

print("Test shape:")
print(X_test.shape)

# ---------------------------------------------------------
# Test prediction
# ---------------------------------------------------------

print("\nRunning final test evaluation...")

start_time = time.time()

test_predictions = model.predict(
    X_test
)

test_inference_time = time.time() - start_time

# ---------------------------------------------------------
# Test metrics
# ---------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nTEST RESULTS")

print("\nAccuracy:")
print(test_accuracy)

print("\nPrecision:")
print(test_precision)

print("\nRecall:")
print(test_recall)

print("\nF1 Score:")
print(test_f1)

print("\nConfusion Matrix:")
print(test_cm)

if test_cm.shape == (2, 2):

    tn, fp, fn, tp = test_cm.ravel()

    test_fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print("\nFalse Positive Rate:")
    print(test_fpr)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print("\nTest inference time:")
print(
    round(
        test_inference_time,
        2
    ),
    "seconds"
)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

print("\nFeature importance:")

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print("\nRandom Forest model saved to:")
print(MODEL_PATH)

print("\nRandom Forest training completed successfully.")