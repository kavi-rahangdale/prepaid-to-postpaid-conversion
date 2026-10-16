"""Synthetic data generator for the prepaid -> postpaid conversion project.

Run from the project root:
    python -m src.data.generate_data
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
TARGET = "converted_60d"


def load_config(path: Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _calibrate_intercept(score: np.ndarray, target_rate: float) -> float:
    """Binary search for an intercept so that mean probability == target_rate."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(score + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_clean_data(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Create a clean dataset with realistic distributions and relationships."""
    n = cfg["data_generation"]["n_customers"]
    positive_rate = cfg["data_generation"]["positive_rate"]

    # --- Demographics -------------------------------------------------
    age = np.clip(rng.normal(34, 11, n), 18, 75).round().astype(int)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.47, 0.01])
    city_tier = rng.choice([1, 2, 3], n, p=[0.35, 0.35, 0.30])

    # Premium handsets are more common in bigger cities
    p_premium = np.select([city_tier == 1, city_tier == 2], [0.25, 0.15], 0.06)
    u = rng.random(n)
    handset_type = np.where(u < p_premium, "Premium", np.where(u < p_premium + 0.40, "Mid", "Budget"))
    is_premium = (handset_type == "Premium").astype(int)
    is_budget = (handset_type == "Budget").astype(int)

    tenure_months = np.clip(rng.gamma(2.0, 14.0, n), 1, 180).round().astype(int)

    # --- Revenue and recharge behaviour --------------------------------
    handset_effect = np.select([is_premium == 1, is_budget == 0], [0.45, 0.15], -0.20)
    tier_effect = np.select([city_tier == 1, city_tier == 2], [0.15, 0.0], -0.10)
    log_arpu = np.log(150) + handset_effect + tier_effect + rng.normal(0, 0.40, n)
    arpu = np.exp(log_arpu)

    recharge_count = rng.poisson(np.clip(1.5 + arpu / 50, 1, None))
    total_spend = arpu * 3 * rng.lognormal(0, 0.10, n)
    avg_recharge_amount = np.where(recharge_count > 0, total_spend / np.maximum(recharge_count, 1), 0.0)
    days_since_last_recharge = np.clip(
        rng.exponential(90 / (recharge_count + 1)), 0, 365
    ).round().astype(int)

    # --- Usage behaviour ------------------------------------------------
    arpu_gap = log_arpu - np.log(150)
    data_usage_gb = rng.lognormal(np.log(8) + 0.9 * arpu_gap + 0.3 * (age < 30), 0.6)
    voice_minutes = rng.lognormal(np.log(450) + 0.25 * arpu_gap, 0.6)
    sms_count = rng.poisson(np.clip(60 - 1.0 * (age - 18), 5, None))
    roaming_days = rng.poisson(0.4 + 0.5 * is_premium + 0.2 * (city_tier == 1))
    app_logins_30d = rng.poisson(3 + 4 * (age < 35) + 3 * is_premium)
    complaints = rng.poisson(0.30, n)
    postpaid_inquiries = rng.poisson(0.08, n)

    # --- Target: latent propensity -> probability -> coin flip ----------
    score = (
        0.55 * _zscore(log_arpu)
        + 0.45 * _zscore(np.log1p(data_usage_gb))
        + 0.25 * _zscore(np.log1p(tenure_months))
        + 0.35 * _zscore(recharge_count)
        - 0.30 * _zscore(days_since_last_recharge)
        + 0.70 * is_premium
        - 0.30 * is_budget
        + 0.30 * (city_tier == 1)
        - 0.35 * complaints
        + 0.90 * postpaid_inquiries
        + 0.25 * _zscore(app_logins_30d)
        - 0.004 * (age - 35) ** 2
        + rng.normal(0, 1.8, n)  # large noise: real-world signal is weak
    )
    intercept = _calibrate_intercept(score, positive_rate)
    converted = (rng.random(n) < _sigmoid(score + intercept)).astype(int)

    # Operational field written by the back-office when a SIM is migrated
    welcome_kit_sent = np.where(
        converted == 1, rng.random(n) < 0.93, rng.random(n) < 0.01
    ).astype(int)

    snapshot_dates = pd.date_range("2025-01-31", periods=6, freq="ME").to_numpy()

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(1, n + 1)],
            "snapshot_date": pd.to_datetime(rng.choice(snapshot_dates, n)).strftime("%Y-%m-%d"),
            "age": age,
            "gender": gender,
            "city_tier": city_tier,
            "handset_type": handset_type,
            "tenure_months": tenure_months,
            "arpu_90d_avg": arpu.round(2),
            "recharge_count_90d": recharge_count,
            "avg_recharge_amount": avg_recharge_amount.round(2),
            "days_since_last_recharge": days_since_last_recharge,
            "data_usage_gb_90d": data_usage_gb.round(2),
            "voice_minutes_90d": voice_minutes.round(1),
            "sms_count_90d": sms_count,
            "roaming_days_90d": roaming_days,
            "app_logins_30d": app_logins_30d,
            "complaints_90d": complaints,
            "postpaid_inquiry_calls_90d": postpaid_inquiries,
            "welcome_kit_sent": welcome_kit_sent,
            TARGET: converted,
        }
    )


def inject_data_issues(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Make the data messy the way real data is: missing, outliers, invalid, duplicates."""
    df = df.copy()
    n = len(df)
    miss = cfg["data_generation"]["missing_rate"]
    outlier_rate = cfg["data_generation"]["outlier_rate"]

    # 1) Missing values (different rate per column)
    missing_rates = {
        "age": miss * 0.6,
        "city_tier": miss * 0.4,
        "handset_type": miss * 0.5,
        "arpu_90d_avg": miss * 0.8,
        "data_usage_gb_90d": miss * 1.2,
        "voice_minutes_90d": miss * 0.4,
        "app_logins_30d": miss * 1.0,
    }
    for col, rate in missing_rates.items():
        if df[col].dtype.kind in "iu":
            df[col] = df[col].astype("float64")
        df.loc[rng.random(n) < rate, col] = np.nan

    # 2) Outliers: a few customers with extreme usage or revenue
    for col in ["data_usage_gb_90d", "voice_minutes_90d", "arpu_90d_avg"]:
        mask = rng.random(n) < outlier_rate
        df.loc[mask, col] = (df.loc[mask, col] * rng.uniform(5, 15, mask.sum())).round(2)

    # 3) Invalid values (data entry / system errors)
    bad_age = rng.random(n) < 0.002
    df.loc[bad_age, "age"] = rng.choice([0, 5, 120, 150], bad_age.sum())
    bad_tenure = rng.random(n) < 0.001
    df.loc[bad_tenure, "tenure_months"] = -1
    bad_recharge = rng.random(n) < 0.001
    df.loc[bad_recharge, "avg_recharge_amount"] = -abs(df.loc[bad_recharge, "avg_recharge_amount"])

    # 4) Inconsistent categories
    lower = rng.random(n) < 0.05
    df.loc[lower, "gender"] = df.loc[lower, "gender"].str.lower()
    short = rng.random(n) < 0.03
    df.loc[short, "gender"] = df.loc[short, "gender"].replace({"Male": "M", "Female": "F"})

    # 5) Duplicate rows
    dup_idx = rng.choice(n, size=int(n * 0.002), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return df


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    clean = generate_clean_data(cfg, rng)
    raw = inject_data_issues(clean, cfg, rng)

    out_path = ROOT / cfg["paths"]["raw_data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_path, index=False)

    print(f"Saved {len(raw):,} rows x {raw.shape[1]} columns -> {out_path}")
    print(f"Positive rate: {clean[TARGET].mean():.2%}")
    print("Missing values per column:")
    print(raw.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
