
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from src.model_registry import load_active

# 1. Load model
model, meta = load_active()
features = meta["features"]

print("=" * 60)
print("PORTSCAN ATTACK DETECTION EVALUATION")
print("=" * 60)

print("Model:", meta["version"])
print("Algorithm:", meta["algorithm"])

# 2. Load dataset
path = "data/Portscan-Friday-no-metadata.parquet"
df = pd.read_parquet(path)
df.columns = df.columns.str.strip()

print("\nDataset shape:", df.shape)

# 3. Normalize labels
df["Label"] = df["Label"].astype(str).str.strip()

print("\nActual labels:")
print(df["Label"].value_counts())

# 4. Validate features
missing = [f for f in features if f not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

# 5. Prepare features
X = df[features].copy()

for column in features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)

for column in ["Flow Bytes/s", "Flow Packets/s"]:
    if column in X.columns:
        valid &= X[column] >= 0

X = X.loc[valid].astype("float32")
y = df.loc[valid, "Label"]

print("\nValid records:", len(X))
print("Rejected records:", int((~valid).sum()))

# 6. Select up to 3000 samples per class
indices = []

for label in y.unique():
    class_indices = y[y == label].index

    selected = y.loc[class_indices].sample(
        n=min(3000, len(class_indices)),
        random_state=42
    )

    indices.extend(selected.index.tolist())

X_test = X.loc[indices]
y_true = y.loc[indices]

print("\nSelected samples:")
print(y_true.value_counts())

# 7. Predict
print("\nRunning predictions...")
y_pred = model.predict(X_test)

# 8. Classification report
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    zero_division=0
))

# 9. Confusion matrix
labels = sorted(set(y_true) | set(y_pred))

print("\nCONFUSION MATRIX")
print("Labels:", labels)
print(confusion_matrix(
    y_true,
    y_pred,
    labels=labels
))

# 10. Prediction distribution
print("\nPREDICTED CLASS DISTRIBUTION")
print(pd.Series(y_pred).value_counts())

# 11. Summary
print("\nEVALUATION SUMMARY")
print("Model:", meta["version"])
print("Algorithm:", meta["algorithm"])
print("Test records:", len(X_test))
print(
    "Correct predictions:",
    int((y_true.to_numpy() == y_pred).sum())
)

print("\nPortScan evaluation completed.")
