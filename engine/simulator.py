"""
What-If Intervention Simulator.
Allows users to simulate the impact of interventions on funnel conversion
using counterfactual reasoning.
"""

import numpy as np
import pandas as pd
from typing import Optional


STAGE_NAMES = {0: "Landing", 1: "FD View", 2: "Details", 3: "KYC", 4: "Deposit"}
STAGE_KEYS = ["landing", "fd_view", "details", "kyc", "deposit"]

# ── Predefined intervention scenarios ──────────────────────────────────────────

SCENARIOS = {
    "simplify_mobile_kyc": {
        "name": "Simplify KYC for Mobile Users",
        "description": "Switch mobile users to Aadhaar-OTP verification, eliminating document upload",
        "target_filter": {"device_mobile": 1},
        "stage_affected": "kyc",
        "lift_pct": 22.0,
        "cost_per_user": 0.0,
        "implementation_effort": "Medium (2-3 sprint effort)",
    },
    "tier3_trust_banner": {
        "name": "Add DICGC Trust Banner for Tier-3",
        "description": "Display RBI deposit insurance badge and ₹5L guarantee for Tier-3 city users",
        "target_filter": {"city_tier": 3},
        "stage_affected": "details",
        "lift_pct": 11.0,
        "cost_per_user": 0.0,
        "implementation_effort": "Low (1 sprint)",
    },
    "evening_rate_bonus": {
        "name": "Evening Rate Bonus Offer",
        "description": "Offer +0.1% rate bonus during 6-9 PM sessions (time-limited 48hr)",
        "target_filter": {"evening_session_ratio_gt": 0.5},
        "stage_affected": "deposit",
        "lift_pct": 16.0,
        "cost_per_user": 0.0,
        "implementation_effort": "Low (1 sprint)",
    },
    "partner_fast_track": {
        "name": "Partner Referral Fast-Track",
        "description": "Auto-apply welcome rate and skip landing for partner-referred users",
        "target_filter": {"referral_partner": 1},
        "stage_affected": "fd_view",
        "lift_pct": 20.0,
        "cost_per_user": 0.0,
        "implementation_effort": "Medium (2 sprints)",
    },
    "high_value_concierge": {
        "name": "High-Value RM Callback",
        "description": "Assign relationship manager callback for high-income hesitant users",
        "target_filter": {"income_num_gte": 3},
        "stage_affected": "kyc",
        "lift_pct": 25.0,
        "cost_per_user": 50.0,
        "implementation_effort": "High (requires RM team)",
    },
    "reentry_reminder": {
        "name": "Drop-off Re-entry SMS Campaign",
        "description": "Send personalized SMS with FD calculator link to users who dropped at Details/KYC",
        "target_filter": {"max_stage_idx_gte": 2, "converted": 0},
        "stage_affected": "details",
        "lift_pct": 9.0,
        "cost_per_user": 0.5,
        "implementation_effort": "Low (1 sprint)",
    },
}


def _apply_filter(df: pd.DataFrame, target_filter: dict) -> pd.DataFrame:
    """Apply a target filter dictionary to select affected users."""
    mask = pd.Series(True, index=df.index)
    for key, value in target_filter.items():
        if key.endswith("_gt"):
            col = key[:-3]
            if col in df.columns:
                mask &= df[col] > value
        elif key.endswith("_gte"):
            col = key[:-4]
            if col in df.columns:
                mask &= df[col] >= value
        elif key.endswith("_lt"):
            col = key[:-3]
            if col in df.columns:
                mask &= df[col] < value
        else:
            if key in df.columns:
                mask &= df[key] == value
    return df[mask]


def simulate_intervention(
    feature_df: pd.DataFrame,
    funnel_data: dict,
    scenario_key: str,
    avg_fd_amount: float = 50_000,
) -> dict:
    """
    Simulate a what-if scenario.
    Returns baseline vs. intervention comparison.
    """
    if scenario_key not in SCENARIOS:
        return {"error": f"Unknown scenario: {scenario_key}"}

    scenario = SCENARIOS[scenario_key]
    total_users = len(feature_df)

    # Identify affected users
    affected = _apply_filter(feature_df, scenario["target_filter"])
    n_affected = len(affected)

    if n_affected == 0:
        return {"error": "No users match this scenario's target filter"}

    # Baseline funnel values
    baseline_values = funnel_data["values"].copy()
    baseline_stages = funnel_data["stages"].copy()
    stage_keys = funnel_data.get("stage_keys", STAGE_KEYS)

    # Find stage index for intervention
    affected_stage = scenario["stage_affected"]
    if affected_stage in stage_keys:
        stage_idx = stage_keys.index(affected_stage)
    else:
        stage_idx = 2  # default to details

    # Calculate intervention impact
    lift_pct = scenario["lift_pct"] / 100
    affected_pct = n_affected / max(total_users, 1)

    # New funnel values after intervention
    intervention_values = baseline_values.copy()
    for i in range(stage_idx, len(intervention_values)):
        # The lift applies to the affected users' progression from the target stage onward
        current_drop = 1 - (intervention_values[i] / max(intervention_values[max(i-1, 0)], 1))
        reduced_drop = current_drop * (1 - lift_pct * affected_pct)
        if i > 0:
            intervention_values[i] = int(
                intervention_values[i-1] * (1 - reduced_drop)
            )

    # Ensure intervention values don't exceed previous stage or go below baseline somehow
    for i in range(1, len(intervention_values)):
        intervention_values[i] = min(intervention_values[i], intervention_values[i-1])
        intervention_values[i] = max(intervention_values[i], baseline_values[i])

    # Additional conversions
    additional_conversions = intervention_values[-1] - baseline_values[-1]
    additional_revenue = additional_conversions * avg_fd_amount

    # Cost
    total_cost = n_affected * scenario["cost_per_user"]
    roi = additional_revenue / max(total_cost, 1) if total_cost > 0 else float("inf")

    # Conversion rate change
    baseline_conv = baseline_values[-1] / max(baseline_values[0], 1) * 100
    intervention_conv = intervention_values[-1] / max(intervention_values[0], 1) * 100

    return {
        "scenario": scenario,
        "scenario_key": scenario_key,
        "baseline_values": baseline_values,
        "intervention_values": intervention_values,
        "stages": baseline_stages,
        "n_affected_users": n_affected,
        "affected_pct": round(affected_pct * 100, 1),
        "additional_conversions": additional_conversions,
        "additional_revenue": additional_revenue,
        "total_cost": total_cost,
        "roi": round(roi, 1) if roi != float("inf") else "∞ (zero cost)",
        "baseline_conversion_rate": round(baseline_conv, 1),
        "intervention_conversion_rate": round(intervention_conv, 1),
        "conversion_lift_pp": round(intervention_conv - baseline_conv, 1),
    }


def simulate_all_scenarios(
    feature_df: pd.DataFrame,
    funnel_data: dict,
    avg_fd_amount: float = 50_000,
) -> pd.DataFrame:
    """
    Run all predefined scenarios and return comparison table.
    """
    rows = []
    for key, scenario in SCENARIOS.items():
        result = simulate_intervention(feature_df, funnel_data, key, avg_fd_amount)
        if "error" not in result:
            rows.append({
                "Scenario": scenario["name"],
                "Users Affected": f"{result['n_affected_users']} ({result['affected_pct']}%)",
                "Additional Conversions": f"+{result['additional_conversions']}",
                "Revenue Impact": f"₹{result['additional_revenue']:,.0f}",
                "Cost": f"₹{result['total_cost']:,.0f}",
                "ROI": f"{result['roi']}×" if isinstance(result['roi'], (int, float)) else result['roi'],
                "Conv. Rate Change": f"{result['baseline_conversion_rate']}% → {result['intervention_conversion_rate']}%",
                "Effort": scenario["implementation_effort"],
            })
    return pd.DataFrame(rows)
