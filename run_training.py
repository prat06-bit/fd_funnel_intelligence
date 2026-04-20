from models.train_models import train_and_save
import pandas as pd

# ---- LOAD DATA ----
feature_df = pd.read_csv("data/fd_data_with_segments.csv")
events_df = pd.read_csv("data/fd_raw.csv")

print("Feature shape:", feature_df.shape)
print("Events shape:", events_df.shape)

# ---- RUN TRAINING ----
results = train_and_save(feature_df, events_df)

# ---- PRINT FINAL METRICS ----
print("\n===== FINAL METRICS =====")
for k, v in results["metrics"].items():
    print(f"{k}: {v}")

print("\n===== MODEL COMPARISON =====")
print(results["model_comparison"])