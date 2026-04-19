import os
from openai import OpenAI

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"


def _call_nvidia(prompt: str, max_tokens: int = 350) -> str:
    if not NVIDIA_API_KEY:
        return " NVIDIA_API_KEY not set. Set it in your .env or run: set NVIDIA_API_KEY=your-key"

    try:
        client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
        )

        completion = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_tokens,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f" NVIDIA API error: {str(e)}"


#  Decision Engine Insights 
def intervention_explanation(user_profile: dict, intervention_plan: dict) -> str:
    interventions_text = ""
    for i, intv in enumerate(intervention_plan.get("interventions", [])[:3], 1):
        interventions_text += f"\n  {i}. [{intv['intervention_type'].upper()}] {intv['action']} (Channel: {intv['channel']}, Expected lift: +{intv['expected_lift_pct']}%)"

    prompt = f"""You are a senior fintech growth strategist at Blostem, an FD platform backed by Zerodha.

The decision engine has analyzed a user and produced the following intervention plan:

User Profile:
- Age: {user_profile.get('age', 'N/A')} | City Tier: {user_profile.get('city_tier', 'N/A')} | Device: {user_profile.get('device_type', 'N/A')}
- Income: {user_profile.get('income_bracket', 'N/A')} | Referral: {user_profile.get('referral_source', 'N/A')}
- Deepest funnel stage reached: {intervention_plan.get('predicted_drop_stage', 'N/A')}
- Total funnel time: {user_profile.get('total_time_in_funnel', 0):.0f} seconds
- KYC attempts: {user_profile.get('kyc_attempts', 0)}

ML Predictions:
- Conversion probability: {intervention_plan.get('conversion_probability', 0)*100:.1f}%
- Risk level: {intervention_plan.get('risk_level', 'N/A').upper()}
- Segment: {intervention_plan.get('segment', 'N/A')}
- Revenue at risk: ₹{intervention_plan.get('estimated_revenue_at_risk', 0):,.0f}

Recommended Interventions:{interventions_text}

Write exactly 3 sentences:
1. Why this specific user is at risk based on their behavioral pattern (not just demographics).
2. Which of the recommended interventions will have the highest impact and why.
3. The specific business metric this intervention will move and by how much.

Be data-driven, specific, and actionable. Use numbers."""

    return _call_nvidia(prompt, max_tokens=300)


def segment_insight(profile: dict) -> str:
    prompt = f"""You are a senior fintech analyst at Blostem, a Fixed Deposit platform.
Analyze this behavioral segment and produce a concise business briefing.

Segment: {profile.get('label', 'Unknown')}
- Users: {profile['size']:,}
- Conversion rate: {profile['conversion_rate']*100:.1f}%
- Avg funnel events: {profile['avg_events']:.1f}
- Avg time in funnel: {profile['avg_time_in_funnel']:.0f} seconds
- Avg scroll depth: {profile['avg_scroll_depth']:.1f}%
- Avg KYC attempts: {profile['avg_kyc_attempts']:.2f}
- Mobile %: {profile['mobile_pct']*100:.0f}%
- Re-entry rate: {profile['reentry_rate']*100:.0f}%

Write exactly 3 sentences:
1. Who these users are behaviorally — what defines their funnel journey pattern.
2. The key business risk or opportunity this segment represents in revenue terms.
3. One specific, actionable intervention Blostem should deploy in the next 2 weeks, with expected conversion lift.

Be specific. Use numbers. No bullet points."""

    return _call_nvidia(prompt, max_tokens=280)


def portfolio_summary(metrics: dict, health_score: dict, n_users: int,
                      model_comparison: dict = None) -> str:
    comparison_text = ""
    if model_comparison:
        comparison_text = f"""
Model Comparison:
- Baseline ({model_comparison['baseline']['name']}): AUC {model_comparison['baseline']['auc']}
- Primary ({model_comparison['primary']['name']}): AUC {model_comparison['primary']['auc']}
- Improvement: +{model_comparison['improvement_auc']}% AUC"""

    prompt = f"""You are the Head of Analytics at Blostem, an FD platform.
Write a 4-sentence executive summary for the leadership team.

Portfolio overview:
- Total users tracked: {n_users:,}
- Funnel conversion rate: {health_score.get('conversion_rate', 0)}%
- Funnel Health Score: {health_score.get('overall_score', 0)}/100
- Worst bottleneck: {health_score.get('worst_bottleneck', 'N/A')} stage ({health_score.get('worst_drop_rate', 0)}% drop-off)
- Model AUC: {metrics['auc']} | F1: {metrics['f1']} | CV AUC: {metrics.get('cv_auc_mean', 'N/A')} ± {metrics.get('cv_auc_std', 'N/A')}
{comparison_text}

Sentence 1: State of the funnel in plain terms — what percentage of revenue is being lost.
Sentence 2: What the ML models tell us about our prediction confidence and where the bottleneck is.
Sentence 3: The single highest-ROI intervention we should deploy this week.
Sentence 4: The expected revenue impact if we act on the decision engine's recommendations.

Executive tone. Numbers-first. No jargon."""

    return _call_nvidia(prompt, max_tokens=350)


def whatif_narrative(simulation_result: dict) -> str:
    scenario = simulation_result.get("scenario", {})

    prompt = f"""You are a growth strategist presenting a what-if analysis to stakeholders.

Scenario: {scenario.get('name', 'N/A')}
Description: {scenario.get('description', 'N/A')}

Results:
- Users affected: {simulation_result.get('n_affected_users', 0)} ({simulation_result.get('affected_pct', 0)}% of total)
- Additional conversions: +{simulation_result.get('additional_conversions', 0)}
- Revenue impact: ₹{simulation_result.get('additional_revenue', 0):,.0f}
- Implementation cost: ₹{simulation_result.get('total_cost', 0):,.0f}
- ROI: {simulation_result.get('roi', 'N/A')}×
- Conversion rate: {simulation_result.get('baseline_conversion_rate', 0)}% → {simulation_result.get('intervention_conversion_rate', 0)}%
- Implementation effort: {scenario.get('implementation_effort', 'N/A')}

Write exactly 2 sentences:
1. Quantify the business case: revenue, conversions, and ROI in concrete terms.
2. State the recommended next step with a specific timeline.

Be crisp and executive-ready."""

    return _call_nvidia(prompt, max_tokens=200)