"""
Run this once before launching the app to pre-generate data and train models.
Usage: python setup.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_data import generate_fd_data
from features.pipeline import build_features
from models.train_models import train_and_save

print("=" * 60)
print("  FD Funnel Intelligence Engine — Setup")
print("=" * 60)

# ── Step 1: Generate multi-table data ──────────────────────────────────────────
print("\n[1/3] Generating synthetic funnel dataset (2000 users)...")
data = generate_fd_data(2000, save_dir=os.path.join(ROOT, "data", "raw"))

users = data["users"]
events = data["funnel_events"]
transactions = data["fd_transactions"]

conv_rate = users["converted"].mean()
print(f"      [OK] {len(users)} users | {len(events)} events | {len(transactions)} transactions")
print(f"      >> Conversion rate: {conv_rate:.1%}")

# ── Step 2: Feature engineering ────────────────────────────────────────────────
print("\n[2/3] Engineering features (30+ behavioral/temporal features)...")
feature_df = build_features(users, events, transactions)
print(f"      [OK] {len(feature_df)} rows x {feature_df.shape[1]} columns")

# ── Step 3: Train models ──────────────────────────────────────────────────────
print("\n[3/3] Training models (conversion + drop-off + segmentation + SHAP)...")
results = train_and_save(feature_df, events, model_dir=os.path.join(ROOT, "models", "saved"))

print("\n" + "=" * 60)
print("  [OK]  Setup complete. Run the app with:")
print("      streamlit run app.py")
print("=" * 60)
