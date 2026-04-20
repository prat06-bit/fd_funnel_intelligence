from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

#  Constants 
STAGE_NAMES: dict[int, str] = {
    0: "Landing", 1: "FD View", 2: "Details", 3: "KYC", 4: "Deposit",
}
STAGE_KEYS: list[str] = ["landing", "fd_view", "details", "kyc", "deposit"]

#  Scenario registry 
SCENARIOS: dict[str, dict[str, Any]] = {
    "simplify_mobile_kyc": {
        "name":                  "Simplify KYC for Mobile Users",
        "description":           "Switch mobile users to Aadhaar-OTP verification, eliminating document upload",
        "target_filter":         {"device_mobile": 1},
        "stage_affected":        "kyc",
        "lift_pct":              22.0,
        "cost_per_user":         0.0,
        "implementation_effort": "Medium (2-3 sprints)",
    },
    "tier3_trust_banner": {
        "name":                  "Add DICGC Trust Banner for Tier-3",
        "description":           "Display RBI deposit insurance badge and ₹5L guarantee for Tier-3 city users",
        "target_filter":         {"city_tier": 3},
        "stage_affected":        "details",
        "lift_pct":              11.0,
        "cost_per_user":         0.0,
        "implementation_effort": "Low (1 sprint)",
    },
    "evening_rate_bonus": {
        "name":                  "Evening Rate Bonus Offer",
        "description":           "Offer +0.1% rate bonus during 6–9 PM sessions (time-limited 48 hr)",
        "target_filter":         {"evening_session_ratio_gt": 0.5},
        "stage_affected":        "deposit",
        "lift_pct":              16.0,
        "cost_per_user":         0.0,
        "implementation_effort": "Low (1 sprint)",
    },
    "partner_fast_track": {
        "name":                  "Partner Referral Fast-Track",
        "description":           "Auto-apply welcome rate and skip landing for partner-referred users",
        "target_filter":         {"referral_partner": 1},
        "stage_affected":        "fd_view",
        "lift_pct":              20.0,
        "cost_per_user":         0.0,
        "implementation_effort": "Medium (2 sprints)",
    },
    "high_value_concierge": {
        "name":                  "High-Value RM Callback",
        "description":           "Assign relationship manager callback for high-income hesitant users",
        "target_filter":         {"income_num_gte": 3},
        "stage_affected":        "kyc",
        "lift_pct":              25.0,
        "cost_per_user":         50.0,
        "implementation_effort": "High (requires RM team)",
    },
    "reentry_reminder": {
        "name":                  "Drop-off Re-entry SMS Campaign",
        "description":           "Send personalised SMS with FD calculator to users who dropped at Details/KYC",
        "target_filter":         {"max_stage_idx_gte": 2, "converted": 0},
        "stage_affected":        "details",
        "lift_pct":              9.0,
        "cost_per_user":         0.5,
        "implementation_effort": "Low (1 sprint)",
    },
}


#  Filter engine 
_SUFFIX_OPS: list[tuple[str, str]] = [
    ("_gte", ">="),
    ("_lte", "<="),
    ("_gt",  ">"),
    ("_lt",  "<"),
]


def _apply_filter(df: pd.DataFrame, target_filter: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    for raw_key, value in target_filter.items():
        # Check suffixes longest-first 
        col, op = raw_key, "=="
        for suffix, operator in _SUFFIX_OPS:
            if raw_key.endswith(suffix):
                col = raw_key[: -len(suffix)]
                op  = operator
                break

        if col not in df.columns:
            raise KeyError(
                f"Filter column '{col}' (from key '{raw_key}') not found in DataFrame. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        if   op == "==": mask &= df[col] == value
        elif op == ">=": mask &= df[col] >= value
        elif op == "<=": mask &= df[col] <= value
        elif op == ">":  mask &= df[col] >  value
        elif op == "<":  mask &= df[col] <  value

    return df[mask]


#  Impact model 
def _compute_intervention_funnel(
    baseline_values: list[int],
    stage_idx:       int,
    lift_pct:        float,
    n_affected:      int,
    n_total:         int,
) -> list[int]:
    values = baseline_values[:]   
    affected_pct = n_affected / max(n_total, 1)
    lift          = lift_pct / 100.0

    for i in range(stage_idx, len(values)):
        additional = int(values[i] * affected_pct * lift)
        values[i]  = min(values[i] + additional, values[i - 1] if i > 0 else values[i])

    # Enforce monotone non-increasing   
    for i in range(1, len(values)):
        values[i] = min(values[i], values[i - 1])

    return values


# Public API 
def simulate_intervention(
    feature_df:    pd.DataFrame,
    funnel_data:   dict,
    scenario_key:  str,
    avg_fd_amount: float = 50_000.0,
) -> dict:
    if scenario_key not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{scenario_key}'. "
            f"Valid keys: {sorted(SCENARIOS.keys())}"
        )

    scenario    = SCENARIOS[scenario_key]
    total_users = len(feature_df)

    affected = _apply_filter(feature_df, scenario["target_filter"])
    n_affected = len(affected)

    if n_affected == 0:
        return {
            "scenario":    scenario,
            "scenario_key": scenario_key,
            "error":       "No users match this scenario's target filter.",
        }

    stage_keys = funnel_data.get("stage_keys", STAGE_KEYS)
    affected_stage = scenario["stage_affected"]

    if affected_stage not in stage_keys:
        raise ValueError(
            f"Scenario '{scenario_key}' targets stage '{affected_stage}' "
            f"which is not present in funnel_data['stage_keys']: {stage_keys}"
        )
    stage_idx = stage_keys.index(affected_stage)

    baseline_values    = list(funnel_data["values"])
    intervention_values = _compute_intervention_funnel(
        baseline_values, stage_idx, scenario["lift_pct"], n_affected, total_users,
    )

    additional_conversions = intervention_values[-1] - baseline_values[-1]
    additional_revenue     = additional_conversions * avg_fd_amount
    total_cost             = n_affected * scenario["cost_per_user"]

    # ROI
    if total_cost == 0:
        roi: float | str = float("inf")
        roi_display      = "∞ (zero cost)"
    else:
        roi         = round(additional_revenue / total_cost, 1)
        roi_display = roi

    baseline_conv     = baseline_values[-1]     / max(baseline_values[0], 1)     * 100
    intervention_conv = intervention_values[-1] / max(intervention_values[0], 1) * 100

    return {
        "scenario":                    scenario,
        "scenario_key":                scenario_key,
        "baseline_values":             baseline_values,
        "intervention_values":         intervention_values,
        "stages":                      funnel_data["stages"],
        "n_affected_users":            n_affected,
        "affected_pct":                round(n_affected / max(total_users, 1) * 100, 1),
        "additional_conversions":      additional_conversions,
        "additional_revenue":          additional_revenue,
        "total_cost":                  total_cost,
        "roi":                         roi,
        "roi_display":                 roi_display,
        "baseline_conversion_rate":    round(baseline_conv, 2),
        "intervention_conversion_rate":round(intervention_conv, 2),
        "conversion_lift_pp":          round(intervention_conv - baseline_conv, 2),
    }


def simulate_all_scenarios(
    feature_df:    pd.DataFrame,
    funnel_data:   dict,
    avg_fd_amount: float = 50_000.0,
) -> pd.DataFrame:
    rows: list[dict] = []

    for key in SCENARIOS:
        try:
            result = simulate_intervention(feature_df, funnel_data, key, avg_fd_amount)
        except (KeyError, ValueError) as exc:
            warnings.warn(f"Scenario '{key}' skipped: {exc}", stacklevel=2)
            continue

        if "error" in result:
            warnings.warn(f"Scenario '{key}' skipped: {result['error']}", stacklevel=2)
            continue

        rows.append({
            "scenario_key":           key,
            "name":                   result["scenario"]["name"],
            "n_affected_users":       result["n_affected_users"],
            "affected_pct":           result["affected_pct"],
            "additional_conversions": result["additional_conversions"],
            "additional_revenue":     result["additional_revenue"],
            "total_cost":             result["total_cost"],
            "roi":                    result["roi"],
            "baseline_conv_rate":     result["baseline_conversion_rate"],
            "intervention_conv_rate": result["intervention_conversion_rate"],
            "conversion_lift_pp":     result["conversion_lift_pp"],
            "effort":                 result["scenario"]["implementation_effort"],
        })

    return pd.DataFrame(rows)