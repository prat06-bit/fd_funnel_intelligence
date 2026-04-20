from features.pipeline import build_features
import pandas as pd

events_df = pd.read_csv("data/fd_raw.csv")
transactions_df = pd.read_csv("data/transactions.csv")  # ← IF EXISTS

feature_df = build_features(events_df, transactions_df)

print("Features shape:", feature_df.shape)

feature_df.to_csv("data/feature_df.csv", index=False)