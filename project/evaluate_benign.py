
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import config
from src import model_registry as reg

print("=" * 60)
print("BENIGN DATASET EVALUATION")
print("=" * 60)

# Load the existing active model (v2)
model, meta = reg.load_active()


print("Model:", meta.get("version", "v2"))
print("Algorithm:", meta.get("algorithm", "Random Forest"))

# Load Benign Monday dataset
file_path = "data/Benign-Monday-no-metadata.parquet"

df = pd.read_parquet(file_path)

print("\nDataset shape:", df.shape)

# Normalize actual labels
df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)

print("\nActual labels:")
print(df["Label"].value_counts())

# Select expected model features
features = meta["features"]

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

X = df.reindex(columns=features).apply(
    pd.to_numeric, errors="coerce"
)

X = X.replace([np.inf, -np.inf], np.nan)

# Remove invalid records
valid = X.notna().all(axis=1)

for column in ("Flow Bytes/s", "Flow Packets/s"):
    valid &= X[column] >= 0

rejected = int((~valid).sum())

X = X.loc[valid].astype("float32")
y = df.loc[valid, "Label"].astype(str)

print("\nValid records:", len(X))
print("Rejected records:", rejected)

if X.empty:
    raise ValueError("No valid records available for evaluation.")

# Sample up to 10,000 records for manageable evaluation
sample_size = min(10000, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

y_sample = y.loc[X_sample.index]

print("\nSelected samples:")
print(y_sample.value_counts())

# Run predictions using existing model
print("\nRunning predictions...")

predictions = model.predict(X_sample)

# Normalize dataset label formatting
y_sample = (
    y_sample
    .str.replace("\ufffd", "-", regex=False)
    .str.replace(r"\s*-\s*", " - ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_sample,
        predictions,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")

labels = sorted(set(y_sample) | set(predictions))

print("Labels:", labels)

print(
    confusion_matrix(
        y_sample,
        predictions,
        labels=labels
    )
)

print("\nPREDICTED CLASS DISTRIBUTION")
print(pd.Series(predictions).value_counts())

correct = int((y_sample.to_numpy() == predictions).sum())
accuracy = accuracy_score(y_sample, predictions)

print("\nEVALUATION SUMMARY")
print("Model: v2")
print("Algorithm: Random Forest")
print("Test records:", len(X_sample))
print("Correct predictions:", correct)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nBenign dataset evaluation completed.")
