import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

print("starting...")
# Load CSV
df = pd.read_csv("corrected_split_with_pcr.csv")
train_folder = "Your extracted deep encoder training features"
val_folder = "Your extracted deep encoder validation features"

EXPECTED_SHAPE = (256, 7, 24, 24)


def load_and_pool(path):
    patch = np.load(path, allow_pickle=True)  # list of (256, D, H, W)

    return patch.mean(axis=(1, 2, 3))


# Prepare data
X_train, y_train = [], []
X_val, y_val = [], []

for _, row in df.iterrows():
    pid, split, label = row["patient_id"], row["split"], row["pcr"]
    if pd.isna(label) or label == "":
        continue
    filename = f"{pid}.npy"
    path = os.path.join(train_folder if split == "train" else val_folder, filename)
    if not os.path.exists(path):
        continue
    try:
        features = load_and_pool(path)
    except ValueError:
        continue
    if split == "train":
        X_train.append(features)
        y_train.append(label)
    elif split == "test":
        X_val.append(features)
        y_val.append(label)

if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_val, pd.DataFrame):
    X_val = pd.DataFrame(X_val)

X_train.columns = X_train.columns.astype(str)
X_val.columns = X_val.columns.astype(str)


y_train = np.array(y_train)
y_val = np.array(y_val)

print(
    f"len X_train: {len(X_train)}, len y_train: {len(y_train)}, len X_val: {len(X_val)}, len y_val: {len(y_val)}"
)

"""
# Optional: Remove NaNs
train_mask = ~X_train.isna().any(axis=1)
val_mask = ~X_val.isna().any(axis=1)

X_train = X_train.loc[train_mask]
y_train = y_train[train_mask]

X_val = X_val.loc[val_mask]
y_val = y_val[val_mask]

print(
    f"After NaN removal - len X_train: {len(X_train)}, len y_train: {len(y_train)}, len X_val: {len(X_val)}, len y_val: {len(y_val)}"
)
"""
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Train model
model = Pipeline([("clf", XGBClassifier(n_estimators=2, max_depth=5, learning_rate=1))])
model.fit(X_train, y_train)

joblib.dump(model, "test_model.joblib")

print("\nReloading model and running validation...")
loaded_model = joblib.load("test_model.joblib")

# Final prediction
y_proba = loaded_model.predict_proba(X_val)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Report
print(f"ROC AUC Score: {roc_auc_score(y_val, y_proba):.3f}")
print(f"balanced accuracy: {balanced_accuracy_score(y_val, y_pred)}")
print(classification_report(y_val, y_pred))
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
