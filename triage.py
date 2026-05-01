import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import json
import os
import sys

# ── Config ─────────────────────────────────────────────────────
DATA_PATH        = 'data/UNSW_NB15_training-set.csv'
MODEL_PATH       = 'model.pkl'
FEATURES_PATH    = 'expected_features.json'
FEATURES         = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate']
LABEL_COL        = 'label'

# ── Step 1: Load data ───────────────────────────────────────────
print("Loading dataset...")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at '{DATA_PATH}'.")
    print("Make sure UNSW_NB15_training-set.csv is inside a 'data/' folder.")
    sys.exit(1)

try:
    df = pd.read_csv(DATA_PATH)
    # df = df.head(1200)  # uncomment for faster training during development
    print(f"Loaded {len(df):,} rows.")
except Exception as e:
    print(f"ERROR: Failed to read dataset: {e}")
    sys.exit(1)

# ── Step 2: Validate required columns ──────────────────────────
missing_cols = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in dataset: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

# ── Step 3: Handle missing values ──────────────────────────────
null_counts = df[FEATURES].isnull().sum()
if null_counts.any():
    print(f"WARNING: Null values found — filling with 0:\n{null_counts[null_counts > 0]}")
    df[FEATURES] = df[FEATURES].fillna(0)

# ── Step 4: One-hot encode ──────────────────────────────────────
print("Encoding features...")
X = pd.get_dummies(df[FEATURES])
y = df[LABEL_COL]

print(f"Feature matrix shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts().to_string()}")

# ── Step 5: Handle class imbalance ─────────────────────────────
classes      = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
weight_dict  = dict(zip(classes, class_weights))
print(f"Class weights (imbalance correction): {weight_dict}")

# ── Step 6: Train/test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── Step 7: Train model ─────────────────────────────────────────
print("Training RandomForest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',   # handles class imbalance
    n_jobs=-1                  # use all CPU cores
)
model.fit(X_train, y_train)
print("Training complete.")

# ── Step 8: Evaluate ────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))

# ── Step 9: Save model ──────────────────────────────────────────
try:
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")
except Exception as e:
    print(f"ERROR: Could not save model: {e}")
    sys.exit(1)

# ── Step 10: Save expected features ────────────────────────────
# This is critical — the app uses this to align live log columns
# with what the model was trained on, preventing feature mismatch crashes
try:
    expected_features = X_train.columns.tolist()
    with open(FEATURES_PATH, 'w') as f:
        json.dump(expected_features, f)
    print(f"Expected features saved to '{FEATURES_PATH}'")
    print(f"Total features: {len(expected_features)}")
except Exception as e:
    print(f"ERROR: Could not save expected features: {e}")
    sys.exit(1)

print("\nDone. You can now run app.py.")