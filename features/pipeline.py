import numpy as np
import pandas as pd

INCOME_MAP = {"<3L": 1, "3-7L": 2, "7-15L": 3, ">15L": 4}
INCOME_MIDPOINTS = {"<3L": 150_000, "3-7L": 500_000, "7-15L": 1_100_000, ">15L": 2_000_000}
STAGE_ORDER = {"landing": 0, "fd_view": 1, "details": 2, "kyc": 3, "deposit": 4}


def build_features_from_input(input_dict):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([input_dict])

    #  KYC FEATURES 
    df["kyc_avg_time"] = df["time_on_kyc"] / df["kyc_attempts"].replace(0, 1)

    #  TOTAL TIME 
    df["total_time_in_funnel"] = (
        df["time_on_landing"]
        + df["time_on_fd_view"]
        + df["time_on_details"]
        + df["time_on_kyc"]
    )

    #  INTERACTION FEATURES ─
    df["mobile_kyc_friction"] = df["device_mobile"] * df["kyc_attempts"]

    df["high_income_engagement"] = (
        (df["income_num"] >= 3).astype(int)
        * df["total_time_in_funnel"]
    )

    df["tier3_mobile"] = (
        ((df["city_tier"] == 3) & (df["device_mobile"] == 1)).astype(int)
    )

    df["reentry_details_time"] = (
        df["has_reentry"] * df["time_on_details"]
    )

    df["evening_details_depth"] = (
        df["evening_session_ratio"] * df["details_max_scroll"]
    )

    #  DEFAULT VALUES FOR MISSING TRAINING FEATURES 
    defaults = {
        "n_devices_used": 1,
        "device_switched": 0,
        "avg_stage_gap_sec": 25.0,
        "fastest_transition_sec": 10.0,
        "avg_scroll_depth": 50.0,
        "session_hour_mean": 12.0,
        "evening_session_ratio": 0.3,
        "weekday_ratio": 0.6,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    #  FINAL SAFETY 
    from features.pipeline import FEATURE_COLS

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLS]

def build_features(users: pd.DataFrame, events: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()

    #  Demographics encoding
    df["income_num"] = df["income_bracket"].map(INCOME_MAP)
    df["device_mobile"] = (df["device_type"] == "mobile").astype(int)
    df["device_desktop"] = (df["device_type"] == "desktop").astype(int)
    df["referral_paid"] = df["referral_source"].isin(["google_ads", "social_media"]).astype(int)
    df["referral_partner"] = (df["referral_source"] == "partner_referral").astype(int)

    #  Funnel progression features 
    user_max_stage = events.groupby("user_id")["stage"].apply(
        lambda x: max(STAGE_ORDER.get(s, 0) for s in x)
    ).rename("max_stage_idx")
    df = df.merge(user_max_stage, left_on="user_id", right_index=True, how="left")
    df["max_stage_idx"] = df["max_stage_idx"].fillna(0).astype(int)

    # Total events per user
    user_event_count = events.groupby("user_id").size().rename("total_events")
    df = df.merge(user_event_count, left_on="user_id", right_index=True, how="left")
    df["total_events"] = df["total_events"].fillna(0).astype(int)

    # Number of funnel attempts
    user_attempts = events.groupby("user_id")["attempt_number"].max().rename("funnel_attempts_feat")
    df = df.merge(user_attempts, left_on="user_id", right_index=True, how="left")
    df["funnel_attempts_feat"] = df["funnel_attempts_feat"].fillna(1).astype(int)

    # Re-entry: did user come back after dropping?
    df["has_reentry"] = (df["funnel_attempts_feat"] > 1).astype(int)

    # Time-based features    
    user_hour_stats = events.groupby("user_id")["hour_of_day"].agg(
        session_hour_mean="mean",
        session_hour_std="std",
    )
    df = df.merge(user_hour_stats, left_on="user_id", right_index=True, how="left")
    df["session_hour_mean"] = df["session_hour_mean"].fillna(12)
    df["session_hour_std"] = df["session_hour_std"].fillna(0)

    evening_flags = events.copy()
    evening_flags["is_evening"] = evening_flags["hour_of_day"].between(18, 21).astype(int)
    user_evening = evening_flags.groupby("user_id")["is_evening"].mean().rename("evening_session_ratio")
    df = df.merge(user_evening, left_on="user_id", right_index=True, how="left")
    df["evening_session_ratio"] = df["evening_session_ratio"].fillna(0)

    user_weekday = events.groupby("user_id")["is_weekday"].mean().rename("weekday_ratio")
    df = df.merge(user_weekday, left_on="user_id", right_index=True, how="left")
    df["weekday_ratio"] = df["weekday_ratio"].fillna(0.5)

    #  Engagement depth features 
    for stage in ["landing", "fd_view", "details", "kyc"]:
        stage_time = events[events["stage"] == stage].groupby("user_id")[
            "time_on_stage_seconds"
        ].mean().rename(f"time_on_{stage}")
        df = df.merge(stage_time, left_on="user_id", right_index=True, how="left")
        df[f"time_on_{stage}"] = df[f"time_on_{stage}"].fillna(0)

    user_total_time = events.groupby("user_id")["time_on_stage_seconds"].sum().rename("total_time_in_funnel")
    df = df.merge(user_total_time, left_on="user_id", right_index=True, how="left")
    df["total_time_in_funnel"] = df["total_time_in_funnel"].fillna(0)

    user_scroll = events.groupby("user_id")["page_scroll_depth"].mean().rename("avg_scroll_depth")
    df = df.merge(user_scroll, left_on="user_id", right_index=True, how="left")
    df["avg_scroll_depth"] = df["avg_scroll_depth"].fillna(0)

    details_scroll = events[events["stage"] == "details"].groupby("user_id")[
        "page_scroll_depth"
    ].max().rename("details_max_scroll")
    df = df.merge(details_scroll, left_on="user_id", right_index=True, how="left")
    df["details_max_scroll"] = df["details_max_scroll"].fillna(0)

    #  KYC friction features  
    kyc_events = events[events["stage"] == "kyc"]
    kyc_attempts = kyc_events.groupby("user_id").size().rename("kyc_attempts")
    df = df.merge(kyc_attempts, left_on="user_id", right_index=True, how="left")
    df["kyc_attempts"] = df["kyc_attempts"].fillna(0).astype(int)

    # KYC drop: reached kyc but didn't proceed quickly 
    kyc_time = events[events["stage"] == "kyc"].groupby("user_id")["time_on_stage_seconds"].mean().rename("kyc_avg_time")
    df = df.merge(kyc_time, left_on="user_id", right_index=True, how="left")
    df["kyc_avg_time"] = df["kyc_avg_time"].fillna(0)

    #  Device switching 
    user_devices = events.groupby("user_id")["device_type"].nunique().rename("n_devices_used")
    df = df.merge(user_devices, left_on="user_id", right_index=True, how="left")
    df["n_devices_used"] = df["n_devices_used"].fillna(1).astype(int)
    df["device_switched"] = (df["n_devices_used"] > 1).astype(int)

    #  Stage velocity features        
    def _stage_velocity(grp):
        grp = grp.sort_values("timestamp")
        if len(grp) < 2:
            return pd.Series({"avg_stage_gap_sec": 0, "fastest_transition_sec": 0})
        timestamps = grp["timestamp"].values
        gaps = np.diff(timestamps.astype("int64") // 10**9)  # seconds
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            return pd.Series({"avg_stage_gap_sec": 0, "fastest_transition_sec": 0})
        return pd.Series({
            "avg_stage_gap_sec": float(np.mean(gaps)),
            "fastest_transition_sec": float(np.min(gaps)),
        })

    velocity = events.groupby("user_id").apply(_stage_velocity, include_groups=False).reset_index()
    velocity.columns = ["user_id", "avg_stage_gap_sec", "fastest_transition_sec"]
    df = df.merge(velocity, on="user_id", how="left")
    df["avg_stage_gap_sec"] = df["avg_stage_gap_sec"].fillna(0)
    df["fastest_transition_sec"] = df["fastest_transition_sec"].fillna(0)

    #  Financial intent signals   
    if len(transactions) > 0:
        txn_stats = transactions.groupby("user_id").agg(
            n_fds_booked=("transaction_id", "count"),
            avg_fd_amount=("fd_amount", "mean"),
            total_fd_amount=("fd_amount", "sum"),
            avg_tenor=("tenor_months", "mean"),
            avg_rate=("interest_rate", "mean"),
            n_banks=("bank_name", "nunique"),
        )
        df = df.merge(txn_stats, left_on="user_id", right_index=True, how="left")
    else:
        df["n_fds_booked"] = 0
        df["avg_fd_amount"] = 0
        df["total_fd_amount"] = 0
        df["avg_tenor"] = 0
        df["avg_rate"] = 0
        df["n_banks"] = 0

    for col in ["n_fds_booked", "avg_fd_amount", "total_fd_amount", "avg_tenor", "avg_rate", "n_banks"]:
        df[col] = df[col].fillna(0)

    # Amount to income ratio 
    df["income_midpoint"] = df["income_bracket"].map(INCOME_MIDPOINTS)
    df["amount_to_income_ratio"] = np.where(
        df["income_midpoint"] > 0,
        df["avg_fd_amount"] / df["income_midpoint"],
        0,
    )

    #  Interaction features 
    df["mobile_kyc_friction"] = df["device_mobile"] * df["kyc_attempts"]
    df["high_income_engagement"] = (df["income_num"] >= 3).astype(int) * df["total_time_in_funnel"]
    df["tier3_mobile"] = ((df["city_tier"] == 3) & (df["device_mobile"] == 1)).astype(int)
    df["reentry_details_time"] = df["has_reentry"] * df["time_on_details"]
    df["evening_details_depth"] = df["evening_session_ratio"] * df["details_max_scroll"]

    return df

#  Feature column list for ML 

FEATURE_COLS = [
    # Demographics
    "age", "city_tier", "income_num", "device_mobile", "device_desktop",
    "referral_paid", "referral_partner",
    # Funnel re-entry behavior (pre-outcome signals)
    "funnel_attempts_feat", "has_reentry",
    # Temporal
    "session_hour_mean", "evening_session_ratio", "weekday_ratio",
    # Engagement depth (all measured during session, before outcome)
    "time_on_landing", "time_on_fd_view", "time_on_details", "time_on_kyc",
    "total_time_in_funnel", "avg_scroll_depth", "details_max_scroll",
    # KYC friction signals
    "kyc_attempts", "kyc_avg_time",
    # Device
    "n_devices_used", "device_switched",
    # Velocity
    "avg_stage_gap_sec", "fastest_transition_sec",
    # Interactions
    "mobile_kyc_friction", "high_income_engagement", "tier3_mobile",
    "reentry_details_time", "evening_details_depth",
]

TARGET_COL = "converted"


# Pretty labels for UI display
FEATURE_LABELS = {
    "age": "Age",
    "city_tier": "City Tier",
    "income_num": "Income Level",
    "device_mobile": "Mobile Device",
    "device_desktop": "Desktop Device",
    "referral_paid": "Paid Acquisition",
    "referral_partner": "Partner Referral",
    "max_stage_idx": "Deepest Funnel Stage",
    "total_events": "Total Funnel Events",
    "funnel_attempts_feat": "Funnel Re-entries",
    "has_reentry": "Returned After Drop",
    "session_hour_mean": "Avg Session Hour",
    "evening_session_ratio": "Evening Session %",
    "weekday_ratio": "Weekday Session %",
    "time_on_landing": "Time on Landing (s)",
    "time_on_fd_view": "Time on FD View (s)",
    "time_on_details": "Time on Details (s)",
    "time_on_kyc": "Time on KYC (s)",
    "total_time_in_funnel": "Total Funnel Time (s)",
    "avg_scroll_depth": "Avg Scroll Depth %",
    "details_max_scroll": "Details Max Scroll %",
    "kyc_attempts": "KYC Attempts",
    "kyc_avg_time": "Avg KYC Time (s)",
    "n_devices_used": "Devices Used",
    "device_switched": "Switched Devices",
    "avg_stage_gap_sec": "Avg Stage Gap (s)",
    "fastest_transition_sec": "Fastest Transition (s)",
    "mobile_kyc_friction": "Mobile × KYC Attempts",
    "high_income_engagement": "High-Income Engagement",
    "tier3_mobile": "Tier-3 Mobile User",
    "reentry_details_time": "Re-entry × Details Time",
    "evening_details_depth": "Evening × Details Scroll",
}
