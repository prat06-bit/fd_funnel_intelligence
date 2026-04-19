"""
Multi-table synthetic FD funnel data generator.
Produces three linked tables:
  1. users        — demographics + registration metadata
  2. funnel_events — timestamped stage-level events per user
  3. fd_transactions — completed FD bookings (converted users only)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────

FUNNEL_STAGES = ["landing", "fd_view", "details", "kyc", "deposit"]
STAGE_ORDER = {s: i for i, s in enumerate(FUNNEL_STAGES)}
DEVICES = ["mobile", "desktop", "tablet"]
REFERRALS = ["organic", "google_ads", "partner_referral", "social_media", "direct"]
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "IndusInd", "Yes Bank", "IDFC First"]
INCOME_MAP = {"<3L": 1, "3-7L": 2, "7-15L": 3, ">15L": 4}
INCOME_MIDPOINTS = {"<3L": 150_000, "3-7L": 500_000, "7-15L": 1_100_000, ">15L": 2_000_000}


def _generate_users(n: int) -> pd.DataFrame:
    """Generate user demographics table."""
    base_date = datetime(2025, 6, 1)

    city_tiers = np.random.choice([1, 2, 3], n, p=[0.30, 0.40, 0.30])
    ages = np.clip(np.random.normal(38, 12, n), 22, 70).astype(int)
    income_brackets = np.random.choice(
        ["<3L", "3-7L", "7-15L", ">15L"], n, p=[0.25, 0.35, 0.25, 0.15]
    )
    devices = np.random.choice(DEVICES, n, p=[0.55, 0.35, 0.10])
    referrals = np.random.choice(REFERRALS, n, p=[0.20, 0.30, 0.15, 0.20, 0.15])
    platforms = np.random.choice([f"PLT_{i:02d}" for i in range(1, 16)], n)

    reg_offsets = np.clip(np.random.exponential(120, n), 1, 365).astype(int)
    reg_dates = [base_date - timedelta(days=int(d)) for d in reg_offsets]

    return pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(1, n + 1)],
        "age": ages,
        "city_tier": city_tiers,
        "income_bracket": income_brackets,
        "device_type": devices,
        "referral_source": referrals,
        "platform_id": platforms,
        "registration_date": reg_dates,
    })


def _simulate_funnel(users: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate realistic funnel events for each user.
    Conversion probabilities at each stage depend on user attributes
    with non-linear interaction effects.
    """
    events = []
    base_time = datetime(2025, 7, 1, 8, 0, 0)

    for _, u in users.iterrows():
        uid = u["user_id"]
        age = u["age"]
        tier = u["city_tier"]
        income = u["income_bracket"]
        device = u["device_type"]
        referral = u["referral_source"]
        inc_num = INCOME_MAP[income]

        # Base transition probabilities per stage
        # landing → fd_view → details → kyc → deposit
        probs = [0.78, 0.64, 0.62, 0.60]

        # ── Modifiers (non-linear, interaction effects) ────────────────────
        # Mobile users struggle at KYC (form friction)
        if device == "mobile":
            probs[2] -= 0.06  # details → kyc
            probs[3] -= 0.12  # kyc → deposit  (big drop)

        # Tier 3 cities have lower overall conversion
        if tier == 3:
            probs = [p - 0.05 for p in probs]
        elif tier == 1:
            probs = [p + 0.03 for p in probs]

        # Higher income → higher conversion (more financially confident)
        if inc_num >= 3:
            probs[2] += 0.06
            probs[3] += 0.08
        elif inc_num == 1:
            probs[2] -= 0.04
            probs[3] -= 0.06

        # Younger users drop more (less FD awareness)
        if age < 28:
            probs[0] -= 0.08
            probs[1] -= 0.05

        # Google ads users = higher initial intent
        if referral == "google_ads":
            probs[0] += 0.06
            probs[1] += 0.04

        # Partner referrals = trust boost at KYC
        if referral == "partner_referral":
            probs[2] += 0.05
            probs[3] += 0.07

        # Interaction: mobile + tier3 + low income → compounded friction
        if device == "mobile" and tier == 3 and inc_num <= 2:
            probs[3] -= 0.10

        # Clip probabilities
        probs = [np.clip(p, 0.08, 0.95) for p in probs]

        # ── Simulate stage progression ────────────────────────────────────
        # Random session start time
        session_offset = timedelta(
            days=int(np.random.randint(0, 30)),
            hours=int(np.random.choice([9, 10, 11, 14, 15, 18, 19, 20, 21])),
            minutes=int(np.random.randint(0, 60)),
        )
        ts = base_time + session_offset
        hour = ts.hour
        is_weekday = ts.weekday() < 5

        # Evening sessions convert slightly better
        if 18 <= hour <= 21:
            probs = [p + 0.04 for p in probs]
            probs = [np.clip(p, 0.08, 0.95) for p in probs]

        # Number of funnel attempts (some users re-enter)
        n_attempts = int(np.random.choice([1, 2, 3], p=[0.65, 0.25, 0.10]))
        max_stage_reached = 0
        # ── Realistic noise: some users defy their probability ──────────────
        # ~7% of users: high-intent but life events cause abandonment
        # ~4% of users: low-intent but impulse-convert
        noise_roll = np.random.random()
        intent_noise = noise_roll < 0.07  # forces early drop even for high-intent
        impulse_noise = noise_roll > 0.96  # forces completion even for low-intent

        for attempt in range(n_attempts):
            attempt_ts = ts + timedelta(
                days=int(attempt * np.random.randint(1, 5)),
                hours=int(np.random.randint(0, 4)),
            )

            for stage_idx, stage in enumerate(FUNNEL_STAGES):
                # Already in landing — always record
                if stage_idx == 0:
                    time_on_stage = max(5, int(np.random.exponential(25)))
                    scroll_depth = np.clip(np.random.beta(2, 3) * 100, 5, 100)
                else:
                    # Check if user proceeds from previous stage
                    effective_prob = probs[stage_idx - 1]
                    # Apply noise: force drop for intent_noise users at KYC/deposit
                    if intent_noise and stage_idx >= 3:
                        effective_prob = min(effective_prob, 0.15)
                    # Apply noise: let impulse_noise users push through details/kyc
                    if impulse_noise and stage_idx in [2, 3]:
                        effective_prob = max(effective_prob, 0.80)
                    if np.random.random() > effective_prob:
                        break  # dropped off

                    time_on_stage = max(3, int(np.random.exponential(
                        {
                            "fd_view": 35,
                            "details": 90,
                            "kyc": 120,
                            "deposit": 60,
                        }.get(stage, 30)
                    )))
                    scroll_depth = np.clip(np.random.beta(3, 2) * 100, 10, 100)

                    # Users who spend more time on details = higher intent
                    if stage == "details" and time_on_stage > 120:
                        probs[2] = min(probs[2] + 0.05, 0.95)
                        probs[3] = min(probs[3] + 0.05, 0.95)

                event_ts = attempt_ts + timedelta(seconds=int(sum(
                    np.random.randint(5, 30) for _ in range(stage_idx)
                )))

                events.append({
                    "user_id": uid,
                    "timestamp": event_ts,
                    "stage": stage,
                    "attempt_number": attempt + 1,
                    "time_on_stage_seconds": time_on_stage,
                    "page_scroll_depth": round(scroll_depth, 1),
                    "device_type": device if attempt == 0 else np.random.choice(
                        [device, np.random.choice(DEVICES)], p=[0.85, 0.15]
                    ),
                    "hour_of_day": event_ts.hour,
                    "is_weekday": int(event_ts.weekday() < 5),
                    "referral_source": referral,
                })
                max_stage_reached = max(max_stage_reached, stage_idx)

    return pd.DataFrame(events)


def _generate_transactions(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Generate FD transactions for users who reached the deposit stage."""
    converted = events[events["stage"] == "deposit"]["user_id"].unique()
    txns = []

    for uid in converted:
        u = users[users["user_id"] == uid].iloc[0]
        inc_num = INCOME_MAP[u["income_bracket"]]
        n_fds = int(np.random.choice([1, 2, 3, 4, 5], p=[0.40, 0.30, 0.15, 0.10, 0.05]))

        for fd_i in range(n_fds):
            amount = int(np.clip(
                np.random.lognormal(10.5 + inc_num * 0.3, 0.6),
                10_000, 2_000_000
            ))
            tenor = np.random.choice([3, 6, 12, 24, 36], p=[0.08, 0.18, 0.42, 0.22, 0.10])
            rate = round(np.random.uniform(6.5, 9.0), 2)
            bank = np.random.choice(BANKS)
            deposit_event = events[
                (events["user_id"] == uid) & (events["stage"] == "deposit")
            ].iloc[0]
            booking_ts = deposit_event["timestamp"] + timedelta(
                minutes=int(np.random.randint(1, 30)),
                days=int(fd_i * np.random.randint(0, 10)),
            )

            txns.append({
                "transaction_id": f"TXN{len(txns)+1:06d}",
                "user_id": uid,
                "fd_amount": amount,
                "tenor_months": tenor,
                "interest_rate": rate,
                "bank_name": bank,
                "booking_timestamp": booking_ts,
                "maturity_date": booking_ts + timedelta(days=int(tenor) * 30),
            })

    return pd.DataFrame(txns)


def generate_fd_data(n: int = 2000, save_dir: str = None) -> dict:
    """
    Generate the full multi-table dataset.
    Returns dict of DataFrames: {users, funnel_events, fd_transactions}.
    """
    print(f"  Generating {n} users...")
    users = _generate_users(n)

    print(f"  Simulating funnel events...")
    events = _simulate_funnel(users)

    print(f"  Generating FD transactions...")
    transactions = _generate_transactions(users, events)

    # Compute summary stats for users table
    converted_users = set(events[events["stage"] == "deposit"]["user_id"].unique())
    users["converted"] = users["user_id"].isin(converted_users).astype(int)

    max_stages = events.groupby("user_id")["stage"].apply(
        lambda x: max(STAGE_ORDER[s] for s in x)
    )
    users["max_stage_reached"] = users["user_id"].map(max_stages).fillna(0).astype(int)
    users["max_stage_name"] = users["max_stage_reached"].map(
        {i: s for s, i in STAGE_ORDER.items()}
    )

    n_attempts = events.groupby("user_id")["attempt_number"].max()
    users["funnel_attempts"] = users["user_id"].map(n_attempts).fillna(1).astype(int)

    data = {
        "users": users,
        "funnel_events": events,
        "fd_transactions": transactions,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        users.to_csv(os.path.join(save_dir, "users.csv"), index=False)
        events.to_csv(os.path.join(save_dir, "funnel_events.csv"), index=False)
        transactions.to_csv(os.path.join(save_dir, "fd_transactions.csv"), index=False)
        print(f"  [OK] Saved to {save_dir}/")

    conv_rate = users["converted"].mean()
    print(f"  >> {len(users)} users | {len(events)} events | {len(transactions)} transactions")
    print(f"  >> Conversion rate: {conv_rate:.1%}")

    return data


if __name__ == "__main__":
    data = generate_fd_data(2000, save_dir="data/raw")
    print(f"\nUsers:\n{data['users'].head()}")
    print(f"\nEvents:\n{data['funnel_events'].head()}")
    print(f"\nTransactions:\n{data['fd_transactions'].head()}")
