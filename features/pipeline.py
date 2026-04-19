from __future__ import annotations
import numpy as np
import pandas as pd

#  Constants 
INCOME_MAP = {"<3L": 1, "3-7L": 2, "7-15L": 3, ">15L": 4}
STAGE_ORDER = {"landing": 0, "fd_view": 1, "details": 2, "kyc": 3, "deposit": 4}

INCOME_MIDPOINTS = {"<3L": 150_000, "3-7L": 500_000, "7-15L": 1_100_000, ">15L": 2_000_000}

_STAGE_TIME_COLS = ["landing", "fd_view", "details", "kyc"]


#  Feature definitions 
FEATURE_COLS: list[str] = [
    # Demographics
    "age", "city_tier", "income_num", "device_mobile",
    "referral_paid", "referral_partner",
    # Funnel re-entry
    "funnel_attempts_feat", "has_reentry",
    # Temporal
    "session_hour_mean", "evening_session_ratio", "weekday_ratio",
    # Engagement depth
    "time_on_landing", "time_on_fd_view", "time_on_details", "time_on_kyc",
    "total_time_in_funnel", "avg_scroll_depth", "details_max_scroll",
    # KYC friction
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

FEATURE_LABELS: dict[str, str] = {
    "age":                    "Age",
    "city_tier":              "City Tier",
    "income_num":             "Income Level",
    "device_mobile":          "Mobile Device",
    "referral_paid":          "Paid Acquisition",
    "referral_partner":       "Partner Referral",
    "max_stage_idx":          "Deepest Funnel Stage",
    "total_events":           "Total Funnel Events",
    "funnel_attempts_feat":   "Funnel Re-entries",
    "has_reentry":            "Returned After Drop",
    "session_hour_mean":      "Avg Session Hour",
    "evening_session_ratio":  "Evening Session %",
    "weekday_ratio":          "Weekday Session %",
    "time_on_landing":        "Time on Landing (s)",
    "time_on_fd_view":        "Time on FD View (s)",
    "time_on_details":        "Time on Details (s)",
    "time_on_kyc":            "Time on KYC (s)",
    "total_time_in_funnel":   "Total Funnel Time (s)",
    "avg_scroll_depth":       "Avg Scroll Depth %",
    "details_max_scroll":     "Details Max Scroll %",
    "kyc_attempts":           "KYC Attempts",
    "kyc_avg_time":           "Avg KYC Time (s)",
    "n_devices_used":         "Devices Used",
    "device_switched":        "Switched Devices",
    "avg_stage_gap_sec":      "Avg Stage Gap (s)",
    "fastest_transition_sec": "Fastest Transition (s)",
    "mobile_kyc_friction":    "Mobile × KYC Attempts",
    "high_income_engagement": "High-Income Engagement",
    "tier3_mobile":           "Tier-3 Mobile User",
    "reentry_details_time":   "Re-entry × Details Time",
    "evening_details_depth":  "Evening × Details Scroll",
}


#  Shared interaction builder 
def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mobile_kyc_friction"]    = df["device_mobile"] * df["kyc_attempts"]
    df["high_income_engagement"] = (df["income_num"] >= 3).astype(int) * df["total_time_in_funnel"]
    df["tier3_mobile"]           = ((df["city_tier"] == 3) & (df["device_mobile"] == 1)).astype(int)
    df["reentry_details_time"]   = df["has_reentry"] * df["time_on_details"]
    df["evening_details_depth"]  = df["evening_session_ratio"] * df["details_max_scroll"]
    return df


#  Inference path 
def build_features_from_input(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])
    df["kyc_avg_time"] = df["time_on_kyc"] / df["kyc_attempts"].clip(lower=1)

    df["total_time_in_funnel"] = df[
        [f"time_on_{s}" for s in _STAGE_TIME_COLS]
    ].sum(axis=1)

    _add_interaction_features(df)

    _INFERENCE_DEFAULTS: dict[str, float] = {
        "n_devices_used":         1,
        "device_switched":        0,
        "avg_stage_gap_sec":      25.0,    
        "fastest_transition_sec": 10.0,
        "avg_scroll_depth":       42.0,    
        "session_hour_mean":      14.0,    
        "evening_session_ratio":  0.28,
        "weekday_ratio":          0.62,
    }
    for col, val in _INFERENCE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLS]


#  Training path 
def build_features(
    users:        pd.DataFrame,
    events:       pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    df = users.copy()

    #  Demographics 
    df["income_num"]       = df["income_bracket"].map(INCOME_MAP)
    df["device_mobile"]    = (df["device_type"] == "mobile").astype(int)
    df["referral_paid"]    = df["referral_source"].isin(["google_ads", "social_media"]).astype(int)
    df["referral_partner"] = (df["referral_source"] == "partner_referral").astype(int)

    #  Batch all event aggregations then merge once 
    aggs: dict[str, pd.Series] = {}

    aggs["max_stage_idx"] = (
        events.groupby("user_id")["stage"]
        .apply(lambda x: max(STAGE_ORDER.get(s, 0) for s in x))
    )
    aggs["total_events"]        = events.groupby("user_id").size()
    aggs["funnel_attempts_feat"]= events.groupby("user_id")["attempt_number"].max()
    aggs["session_hour_mean"]   = events.groupby("user_id")["hour_of_day"].mean()
    aggs["evening_session_ratio"] = (
        events.assign(is_evening=events["hour_of_day"].between(18, 21).astype(int))
        .groupby("user_id")["is_evening"].mean()
    )
    aggs["weekday_ratio"]       = events.groupby("user_id")["is_weekday"].mean()
    aggs["total_time_in_funnel"]= events.groupby("user_id")["time_on_stage_seconds"].sum()
    aggs["avg_scroll_depth"]    = events.groupby("user_id")["page_scroll_depth"].mean()
    aggs["n_devices_used"]      = events.groupby("user_id")["device_type"].nunique()

    for stage in _STAGE_TIME_COLS:
        aggs[f"time_on_{stage}"] = (
            events[events["stage"] == stage]
            .groupby("user_id")["time_on_stage_seconds"].mean()
        )

    aggs["details_max_scroll"] = (
        events[events["stage"] == "details"]
        .groupby("user_id")["page_scroll_depth"].max()
    )
    aggs["kyc_attempts"] = (
        events[events["stage"] == "kyc"].groupby("user_id").size()
    )
    aggs["kyc_avg_time"] = (
        events[events["stage"] == "kyc"]
        .groupby("user_id")["time_on_stage_seconds"].mean()
    )

    # Stage velocity
    def _stage_velocity(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values("timestamp")
        if len(grp) < 2:
            return pd.Series({"avg_stage_gap_sec": 0.0, "fastest_transition_sec": 0.0})
        ts_sec = grp["timestamp"].values.astype("int64") // 10 ** 9
        gaps   = np.diff(ts_sec)
        gaps   = gaps[gaps > 0]
        if len(gaps) == 0:
            return pd.Series({"avg_stage_gap_sec": 0.0, "fastest_transition_sec": 0.0})
        return pd.Series({
            "avg_stage_gap_sec":      float(np.mean(gaps)),
            "fastest_transition_sec": float(np.min(gaps)),
        })

    _pandas_major = int(pd.__version__.split(".")[0])
    _pandas_minor = int(pd.__version__.split(".")[1])
    _groupby_kwargs = {"include_groups": False} if (_pandas_major, _pandas_minor) >= (2, 2) else {}

    velocity = (
        events.groupby("user_id")
        .apply(_stage_velocity, **_groupby_kwargs)
        .reset_index()
    )
    velocity.columns = ["user_id", "avg_stage_gap_sec", "fastest_transition_sec"]

    #  Single bulk merge 
    agg_df = pd.concat(aggs.values(), axis=1, keys=aggs.keys())
    df = df.merge(agg_df, left_on="user_id", right_index=True, how="left")
    df = df.merge(velocity, on="user_id", how="left")

    #  Fill NaNs with appropriate sentinels 
    _FILLNA: dict[str, float | int] = {
        "max_stage_idx":          0,
        "total_events":           0,
        "funnel_attempts_feat":   1,    # at least 1 attempt
        "session_hour_mean":      12,
        "evening_session_ratio":  0,
        "weekday_ratio":          0.5,
        "total_time_in_funnel":   0,
        "avg_scroll_depth":       0,
        "details_max_scroll":     0,
        "kyc_attempts":           0,
        "kyc_avg_time":           0,
        "n_devices_used":         1,
        "avg_stage_gap_sec":      0,
        "fastest_transition_sec": 0,
        **{f"time_on_{s}": 0 for s in _STAGE_TIME_COLS},
    }
    df = df.fillna(_FILLNA)

    # Integer columns
    for col in ["max_stage_idx", "total_events", "funnel_attempts_feat",
                "kyc_attempts", "n_devices_used"]:
        df[col] = df[col].astype(int)

    df["has_reentry"]     = (df["funnel_attempts_feat"] > 1).astype(int)
    df["device_switched"] = (df["n_devices_used"] > 1).astype(int)

    #  Financial features 
    _TXN_COLS = ["n_fds_booked", "avg_fd_amount", "total_fd_amount",
                 "avg_tenor", "avg_rate", "n_banks"]
    if not transactions.empty:
        txn_stats = transactions.groupby("user_id").agg(
            n_fds_booked  =("transaction_id", "count"),
            avg_fd_amount =("fd_amount",       "mean"),
            total_fd_amount=("fd_amount",      "sum"),
            avg_tenor     =("tenor_months",    "mean"),
            avg_rate      =("interest_rate",   "mean"),
            n_banks       =("bank_name",       "nunique"),
        )
        df = df.merge(txn_stats, left_on="user_id", right_index=True, how="left")

    for col in _TXN_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    _add_interaction_features(df)

    return df