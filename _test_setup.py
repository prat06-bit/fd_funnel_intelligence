"""Quick test of full pipeline"""
import traceback, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.generate_data import generate_fd_data
    from features.pipeline import build_features
    from models.train_models import train_and_save

    # Clean old artifacts
    import shutil
    saved = os.path.join("models", "saved")
    if os.path.exists(saved):
        shutil.rmtree(saved)

    data = generate_fd_data(2000, save_dir="data/raw")
    df = build_features(data["users"], data["funnel_events"], data["fd_transactions"])
    r = train_and_save(df, data["funnel_events"], model_dir="models/saved")
    print("\n\n=== PIPELINE SUCCESS ===")
    print(f"Metrics: {r['metrics']}")
    print(f"Model comparison: {r['model_comparison']}")
    print(f"Funnel: {r['funnel_data']['stages']}")
    print(f"Funnel values: {r['funnel_data']['values']}")
    print(f"Drop rates: {r['drop_rates']}")
    sys.exit(0)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
