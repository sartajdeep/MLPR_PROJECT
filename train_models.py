"""
Train XGBoost and Random Forest Regressors using:
  - 768-dim sentence embeddings from all-mpnet-base-v2 (text column)
  - 28 engineered features (columns J–AK: char_len … noun_count)
to predict target_score (column I).
"""

import numpy as np
import pandas as pd
import os, pickle, warnings
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ────────────────────────────── 1. Load Data ──────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "BABE_features_normalized.csv")
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset  ->  {df.shape[0]} rows, {df.shape[1]} columns")

# Drop rows where target or text is missing
df = df.dropna(subset=["target_score", "text"]).reset_index(drop=True)
print(f"[INFO] After dropping NaN  ->  {df.shape[0]} rows")

# ────────────────────────────── 2. Sentence Embeddings ──────────────────────────────
EMBEDDING_CACHE = os.path.join(os.path.dirname(__file__), "embeddings_mpnet.npy")

if os.path.exists(EMBEDDING_CACHE):
    print("[INFO] Loading cached embeddings …")
    embeddings = np.load(EMBEDDING_CACHE)
else:
    print("[INFO] Loading all-mpnet-base-v2 model …")
    model = SentenceTransformer("all-mpnet-base-v2")
    print("[INFO] Encoding texts (this may take a few minutes) …")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(EMBEDDING_CACHE, embeddings)
    print(f"[INFO] Embeddings saved -> {EMBEDDING_CACHE}")

print(f"[INFO] Embedding matrix shape: {embeddings.shape}")  # (N, 768)

# ────────────────────────────── 3. Feature Matrix ──────────────────────────────
# Columns J to AK  (indices 9 to 36 inclusive) -> 28 engineered features
feature_cols = df.columns[9:37].tolist()
print(f"[INFO] Engineered feature columns ({len(feature_cols)}): {feature_cols}")

X_features = df[feature_cols].values.astype(np.float32)

# Combine embeddings + engineered features -> total 768 + 28 = 796 dims
X = np.hstack([embeddings, X_features])
y = df["target_score"].values.astype(np.float32)

print(f"[INFO] Final feature matrix: X={X.shape}, y={y.shape}")

# ────────────────────────────── 4. Train / Test Split ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Train={X_train.shape[0]}  |  Test={X_test.shape[0]}")


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R2    : {r2:.4f}")
    print(f"{'='*50}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ────────────────────────────── 5. XGBoost Regressor ──────────────────────────────
print("\n[TRAIN] XGBoost Regressor …")
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
xgb_preds = xgb_model.predict(X_test)
xgb_metrics = evaluate("XGBoost Regressor", y_test, xgb_preds)

# ────────────────────────────── 6. Random Forest Regressor ──────────────────────────────
print("\n[TRAIN] Random Forest Regressor …")
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_metrics = evaluate("Random Forest Regressor", y_test, rf_preds)

# ────────────────────────────── 7. Save Models ──────────────────────────────
OUT_DIR = os.path.dirname(__file__)

xgb_path = os.path.join(OUT_DIR, "xgb_regressor.pkl")
rf_path = os.path.join(OUT_DIR, "rf_regressor.pkl")

with open(xgb_path, "wb") as f:
    pickle.dump(xgb_model, f)
with open(rf_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"\n[SAVED] XGBoost  -> {xgb_path}")
print(f"[SAVED] Random Forest -> {rf_path}")

# ────────────────────────────── 8. Comparison Summary ──────────────────────────────
print("\n\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame(
    {"XGBoost": xgb_metrics, "RandomForest": rf_metrics}
).T
print(summary.to_string())
print("=" * 60)

# ────────────────────────────── 9. Plot Actual vs Predicted ──────────────────────────────
plt.figure(figsize=(12, 5))

# XGBoost Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, xgb_preds, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"XGBoost Regressor\nRMSE: {xgb_metrics['RMSE']:.4f}")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")

# Random Forest Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_preds, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"Random Forest Regressor\nRMSE: {rf_metrics['RMSE']:.4f}")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "actual_vs_predicted.png")
plt.savefig(plot_path)
print(f"\n[SAVED] Plot -> {plot_path}")
