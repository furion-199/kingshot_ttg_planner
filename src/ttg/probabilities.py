from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@lru_cache(maxsize=1)
def load_probability_table() -> pl.DataFrame:
    """Load the refinement probability table from CSV."""
    path = DATA_DIR / "refinement_probabilities.csv"
    df = pl.read_csv(path)

    required_cols = {"tier", "yield", "probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df = (
        df.with_columns(
            pl.col("tier").cast(pl.Int64),
            pl.col("yield").cast(pl.Int64),
            pl.col("probability").cast(pl.Float64),
        )
        .sort(["tier", "yield"])
    )

    return df


@lru_cache(maxsize=None)
def tier_distribution(tier: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the support and probabilities for a single tier.

    Returns
    -------
    yields : np.ndarray
        TTG yield values
    probs : np.ndarray
        Corresponding probabilities
    """
    df = load_probability_table().filter(pl.col("tier") == tier)

    if df.is_empty():
        raise ValueError(f"No probability data found for tier={tier}")

    yields = df["yield"].to_numpy()
    probs = df["probability"].to_numpy()

    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"Tier {tier} probabilities sum to {probs.sum()}, not 1.0")

    return yields, probs


def get_tier(ref_index: int) -> int:
    """
    Convert 1-indexed refinement position within a week into a refinement tier.
    """
    if ref_index <= 0:
        raise ValueError("ref_index must be >= 1")
    return ((ref_index - 1) // 20) % 5 + 1


def tier_expected_value(tier: int) -> float:
    yields, probs = tier_distribution(tier)
    return float(np.dot(yields, probs))


def tier_min(tier: int) -> int:
    yields, _ = tier_distribution(tier)
    return int(yields.min())


def tier_max(tier: int) -> int:
    yields, _ = tier_distribution(tier)
    return int(yields.max())


def validate_probability_table() -> pl.DataFrame:
    """
    Return a validation summary showing probability sums by tier.
    Useful for quick notebook checks.
    """
    df = load_probability_table()

    summary = (
        df.group_by("tier")
        .agg(
            pl.col("probability").sum().alias("prob_sum"),
            pl.col("yield").min().alias("min_yield"),
            pl.col("yield").max().alias("max_yield"),
            pl.len().alias("n_outcomes"),
        )
        .sort("tier")
    )

    return summary