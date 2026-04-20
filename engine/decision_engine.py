from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
STAGE_NAMES: dict[int, str] = {
    0: "Landing", 1: "FD View", 2: "Details", 3: "KYC", 4: "Deposit",
}

_RISK_THRESHOLDS = {"high": 0.25, "medium": 0.50}   # < high → high risk, etc.


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class Intervention:
    """A single recommended action for a user."""
    rule_id:            str
    intervention_type:  str
    action:             str
    channel:            str
    timing:             str
    expected_lift_pct:  float
    cost_per_user:      float
    business_rule:      str   = ""
    priority_score:     float = 0.0   # net ROI in ₹; set by evaluate_user
    expected_new_prob:  float = 0.0   # post-intervention conversion probability
    expected_gain:      float = 0.0   # ₹ expected revenue gain
    net_roi:            float = 0.0   # expected_gain − cost_per_user

    def to_dict(self) -> dict:
        return asdict(self)           # all fields are declared → asdict is complete


@dataclass
class InterventionPlan:
    """Complete decision output for one user."""
    user_id:                 str
    conversion_probability:  float
    predicted_drop_stage:    str
    segment:                 str
    risk_level:              str            # "high" | "medium" | "low"
    interventions:           list[Intervention] = field(default_factory=list)
    estimated_revenue_at_risk: float        = 0.0
    top_risk_factors:        list           = field(default_factory=list)

    def to_dict(self) -> dict:
        # asdict() already recurses into nested dataclasses correctly.
        # No manual re-wrapping needed (and doing so would silently drop
        # any fields not in the nested to_dict() call).
        return asdict(self)


# ── ROI helper — defined before any caller ────────────────────────────────────
def compute_intervention_roi(
    conv_prob:  float,
    lift:       float,      # absolute probability lift (e.g. 0.22)
    user_value: float,      # expected FD amount (₹)
    cost:       float,      # cost per user (₹)
) -> dict:
    """
    Returns the incremental expected value of a single intervention.

    new_prob     = min(conv_prob + lift, 1.0)
    expected_gain = ΔP × user_value
    net_roi       = expected_gain − cost
    """
    new_prob      = min(conv_prob + lift, 1.0)
    expected_gain = (new_prob - conv_prob) * user_value
    net_roi       = expected_gain - cost
    return {
        "new_prob":      new_prob,
        "expected_gain": expected_gain,
        "net_roi":       net_roi,        # was "roi" in original — KeyError on every call
    }


# ── Business rules ─────────────────────────────────────────────────────────────
# Each rule holds a *template* Intervention. evaluate_user copies it with
# dataclasses.replace() and then fills in the ROI fields. The templates
# themselves are never mutated.
def _make_rules() -> list[dict]:
    from dataclasses import replace

    _tmpl = dict(expected_new_prob=0.0, expected_gain=0.0, net_roi=0.0)

    rules = [
        {
            "rule_id":   "KYC_FRICTION_MOBILE",
            "condition": lambda u, p: p.get("predicted_drop_stage") == "KYC"
                                      and u.get("device_mobile", 0) == 1,
            "template":  Intervention(
                rule_id="KYC_FRICTION_MOBILE", intervention_type="simplify",
                action="Switch to Aadhaar-OTP verification (skip document upload on mobile)",
                channel="in_app", timing="immediate",
                expected_lift_pct=22.0, cost_per_user=0.0,
                business_rule="Mobile users at KYC have 2.3× higher drop-off than desktop",
            ),
        },
        {
            "rule_id":   "KYC_RETRY",
            "condition": lambda u, p: u.get("kyc_attempts", 0) >= 2
                                      and u.get("converted", 0) == 0,
            "template":  Intervention(
                rule_id="KYC_RETRY", intervention_type="support",
                action="Trigger assisted KYC flow with pre-filled fields + support chat widget",
                channel="in_app", timing="immediate",
                expected_lift_pct=18.0, cost_per_user=2.0,
                business_rule="Users with 2+ KYC attempts are 3× more likely to convert with assisted flow",
            ),
        },
        {
            "rule_id":   "RATE_COMPARISON_NUDGE",
            "condition": lambda u, p: u.get("time_on_details", 0) > 60
                                      and p.get("conversion_probability", 1.0) < 0.45,
            "template":  Intervention(
                rule_id="RATE_COMPARISON_NUDGE", intervention_type="nudge",
                action="Display personalised rate comparison card highlighting best-match FD by tenor + amount",
                channel="in_app", timing="next_session",
                expected_lift_pct=14.0, cost_per_user=0.0,
                business_rule="Users spending >60s on details are comparison-shopping; targeted card converts 14% more",
            ),
        },
        {
            "rule_id":   "DORMANT_REENGAGEMENT",
            "condition": lambda u, p: u.get("has_reentry", 0) == 0
                                      and u.get("max_stage_idx", 0) >= 2
                                      and p.get("conversion_probability", 1.0) < 0.35,
            "template":  Intervention(
                rule_id="DORMANT_REENGAGEMENT", intervention_type="re-engage",
                action="Send personalised FD maturity calculator showing projected returns for their income bracket",
                channel="email", timing="scheduled",
                expected_lift_pct=9.0, cost_per_user=0.5,
                business_rule="Users who reached Details but never returned show 9% re-engagement with projected returns",
            ),
        },
        {
            "rule_id":   "EVENING_HIGH_INTENT",
            "condition": lambda u, p: u.get("evening_session_ratio", 0) > 0.5
                                      and u.get("max_stage_idx", 0) >= 3,
            "template":  Intervention(
                rule_id="EVENING_HIGH_INTENT", intervention_type="offer",
                action="Present limited-time rate bonus (+0.1% for 48 hrs) during evening session",
                channel="in_app", timing="immediate",
                expected_lift_pct=16.0, cost_per_user=0.0,
                business_rule="Evening sessions signal deliberate intent; time-limited offers convert 16% more",
            ),
        },
        {
            "rule_id":   "TIER3_TRUST_BUILD",
            "condition": lambda u, p: u.get("city_tier", 1) == 3
                                      and p.get("predicted_drop_stage") in ("KYC", "Details"),
            "template":  Intervention(
                rule_id="TIER3_TRUST_BUILD", intervention_type="nudge",
                action="Show RBI insurance badge + '₹5L DICGC guaranteed' trust banner with regional language toggle",
                channel="in_app", timing="immediate",
                expected_lift_pct=11.0, cost_per_user=0.0,
                business_rule="Tier-3 users have 18% lower trust scores; DICGC visibility lifts conversion by 11%",
            ),
        },
        {
            "rule_id":   "PARTNER_REFERRAL_WARMUP",
            "condition": lambda u, p: u.get("referral_partner", 0) == 1
                                      and u.get("max_stage_idx", 0) <= 1,
            "template":  Intervention(
                rule_id="PARTNER_REFERRAL_WARMUP", intervention_type="nudge",
                action="Auto-apply partner welcome rate (+0.15%) and skip landing — direct to FD configurator",
                channel="in_app", timing="immediate",
                expected_lift_pct=20.0, cost_per_user=0.0,
                business_rule="Partner-referred users have pre-existing trust; reduce friction to capture high-intent traffic",
            ),
        },
        {
            "rule_id":   "HIGH_AMOUNT_CONCIERGE",
            "condition": lambda u, p: u.get("income_num", 0) >= 3
                                      and p.get("conversion_probability", 1.0) < 0.40,
            "template":  Intervention(
                rule_id="HIGH_AMOUNT_CONCIERGE", intervention_type="support",
                action="Assign RM callback within 2 hours for personalised FD portfolio consultation",
                channel="call", timing="scheduled",
                expected_lift_pct=25.0, cost_per_user=50.0,
                business_rule="High-income hesitant users respond 25% better to personalised RM outreach",
            ),
        },
    ]
    return rules


# Module-level singleton — built once
BUSINESS_RULES: list[dict] = _make_rules()


# ── Core evaluation ────────────────────────────────────────────────────────────
def evaluate_user(
    user_features: dict,
    ml_predictions: dict,
    avg_fd_amount: float = 50_000.0,
) -> InterventionPlan:
    from dataclasses import replace

    conv_prob      = float(ml_predictions.get("conversion_probability", 0.5))
    drop_stage_idx = int(ml_predictions.get("predicted_drop_stage_idx", 0))
    drop_stage     = STAGE_NAMES.get(drop_stage_idx, "Landing")
    segment        = str(ml_predictions.get("segment", "Unknown"))

    if conv_prob < _RISK_THRESHOLDS["high"]:
        risk = "high"
    elif conv_prob < _RISK_THRESHOLDS["medium"]:
        risk = "medium"
    else:
        risk = "low"

    income_num      = float(user_features.get("income_num", 2))
    est_amount      = avg_fd_amount * (1 + (income_num - 2) * 0.3)
    revenue_at_risk = est_amount * (1 - conv_prob)

    plan = InterventionPlan(
        user_id                  = str(user_features.get("user_id", "UNKNOWN")),
        conversion_probability   = conv_prob,
        predicted_drop_stage     = drop_stage,
        segment                  = segment,
        risk_level               = risk,
        estimated_revenue_at_risk= round(revenue_at_risk, 2),
        top_risk_factors         = ml_predictions.get("top_risk_factors", []),
    )

    preds_with_stage = {**ml_predictions, "predicted_drop_stage": drop_stage}

    for rule in BUSINESS_RULES:
        try:
            if not rule["condition"](user_features, preds_with_stage):
                continue
        except Exception as exc:
            log.warning("Rule '%s' condition raised: %s", rule["rule_id"], exc)
            continue

        template = rule["template"]
        lift     = template.expected_lift_pct / 100.0
        roi_vals = compute_intervention_roi(conv_prob, lift, est_amount, template.cost_per_user)

        # Replace creates a new Intervention without mutating the template
        intervention = replace(
            template,
            priority_score    = round(roi_vals["net_roi"], 2),
            expected_new_prob = round(roi_vals["new_prob"], 4),
            expected_gain     = round(roi_vals["expected_gain"], 2),
            net_roi           = round(roi_vals["net_roi"], 2),
        )
        plan.interventions.append(intervention)

    plan.interventions.sort(key=lambda x: -x.priority_score)
    return plan


# ── Batch evaluation ───────────────────────────────────────────────────────────
def batch_evaluate(
    feature_df:        pd.DataFrame,
    conv_proba:        np.ndarray,
    drop_stages:       Optional[np.ndarray],
    segment_names:     pd.Series,
    shap_top_factors:  Optional[list] = None,
    avg_fd_amount:     float          = 50_000.0,
) -> list[InterventionPlan]:
    """
    Evaluate all users in feature_df. Uses itertuples() (22× faster than
    iterrows() on typical DataFrames) because we only need row values, not
    a full Series with index.
    """
    n       = len(feature_df)
    plans   = []
    columns = list(feature_df.columns)

    for i, row in enumerate(feature_df.itertuples(index=False, name=None)):
        user_features = dict(zip(columns, row))
        ml_predictions = {
            "conversion_probability":  float(conv_proba[i]) if i < len(conv_proba) else 0.5,
            "predicted_drop_stage_idx":int(drop_stages[i]) if drop_stages is not None and i < n else 0,
            "segment":                 str(segment_names.iloc[i]) if i < len(segment_names) else "Unknown",
            "top_risk_factors":        shap_top_factors[i] if shap_top_factors and i < len(shap_top_factors) else [],
        }
        plans.append(evaluate_user(user_features, ml_predictions, avg_fd_amount))

    return plans


# ── Priority queue (raw numerics only) ────────────────────────────────────────
def build_priority_queue(
    plans:  list[InterventionPlan],
    top_n:  int = 50,
) -> pd.DataFrame:
    """
    Return a DataFrame of raw numeric values for high/medium-risk users.
    All formatting (₹ symbols, % signs) is the caller's responsibility
    so the DataFrame can be sorted, filtered, and aggregated properly.
    """
    rows: list[dict] = []

    for plan in plans:
        if plan.risk_level not in ("high", "medium") or not plan.interventions:
            continue

        top = plan.interventions[0]
        rows.append({
            "user_id":              plan.user_id,
            "conversion_prob":      round(plan.conversion_probability, 4),
            "risk_level":           plan.risk_level,
            "drop_stage":           plan.predicted_drop_stage,
            "segment":              plan.segment,
            "top_action":           top.action,
            "channel":              top.channel,
            "expected_lift_pct":    top.expected_lift_pct,
            "cost_per_user":        top.cost_per_user,
            "expected_gain":        top.expected_gain,
            "expected_new_prob":    top.expected_new_prob,
            "revenue_at_risk":      plan.estimated_revenue_at_risk,
            "net_roi":              top.net_roi,       # numeric — sort/filter on this
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("net_roi", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .pipe(lambda df: df.assign(**{"rank": range(1, len(df) + 1)}))
        .set_index("rank")
    )


# ── Funnel health score ────────────────────────────────────────────────────────
def compute_funnel_health_score(
    funnel_data: dict,
    drop_rates:  dict,
    metrics:     dict,
) -> dict:
    values    = funnel_data["values"]
    total     = max(values[0], 1)
    converted = values[-1]
    conv_rate = converted / total

    # Each sub-score is 0–100 with a documented ceiling definition
    conv_score       = min(conv_rate / 0.30 * 100, 100)          # 30% conv = 100
    max_drop         = max(drop_rates.values(), default=0.5)
    smoothness_score = max(0.0, (1 - max_drop / 0.60) * 100)     # 60% single-drop = 0
    auc              = float(metrics.get("auc", 0.5))
    model_score      = max(0.0, (auc - 0.5) / 0.4 * 100)         # AUC 0.9 → 100

    if len(values) >= 2:
        avg_reach   = sum(values[1:]) / (len(values) - 1) / total
        depth_score = min(avg_reach / 0.50 * 100, 100)
    else:
        depth_score = 50.0

    overall = (
        conv_score       * 0.40
        + smoothness_score * 0.25
        + model_score      * 0.20
        + depth_score      * 0.15
    )

    worst_stage = max(drop_rates, key=drop_rates.get) if drop_rates else "unknown"

    return {
        "overall_score":          round(overall, 1),
        "conversion_efficiency":  round(conv_score, 1),
        "flow_smoothness":        round(smoothness_score, 1),
        "model_confidence":       round(model_score, 1),
        "engagement_depth":       round(depth_score, 1),
        "worst_bottleneck":       worst_stage,
        "worst_drop_rate":        round(drop_rates.get(worst_stage, 0) * 100, 1),
        "conversion_rate":        round(conv_rate * 100, 1),
    }