from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@lru_cache(maxsize=1)
def load_cost_table() -> pl.DataFrame:
    """Load refinement cost table."""
    path = DATA_DIR / "refinement_costs.csv"

    df = pl.read_csv(path)

    required = {"tier", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = df.with_columns(
        pl.col("tier").cast(pl.Int64),
        pl.col("cost").cast(pl.Float64),
    ).sort("tier")

    return df


@lru_cache(maxsize=None)
def tier_cost(tier: int) -> float:
    df = load_cost_table().filter(pl.col("tier") == tier)

    if df.is_empty():
        raise ValueError(f"No cost defined for tier={tier}")

    return float(df["cost"][0])


def tier_for_refine(ref_index: int) -> int:
    if ref_index <= 0:
        raise ValueError("ref_index must be >= 1")

    tier = ((ref_index - 1) // 20) + 1

    if tier > 5:
        raise ValueError("ref_index exceeds weekly capacity")

    return tier

def normalize_last_week_days(last_week_days: int | None = 5) -> int:
    """
    Weeks run Monday..Sunday.
    If the event ends early in the final week, last_week_days controls how many
    discounted refines exist in that final week.

    Examples:
        5 -> event ends Friday  -> 5 discounted refines
        7 -> full week          -> 7 discounted refines
    """
    if last_week_days is None:
        return 5

    d = int(last_week_days)
    if d < 1 or d > 7:
        raise ValueError("last_week_days must be between 1 and 7")

    return d

@lru_cache(maxsize=None)
def cost_one_week(refines: int, *, discount_refines: int = 7) -> float:
    if refines < 0 or refines > 100:
        raise ValueError("refines must be between 0 and 100")

    if discount_refines < 0 or discount_refines > 7:
        raise ValueError("discount_refines must be between 0 and 7")

    tier_counts = [0, 0, 0, 0, 0, 0]  # 1-based

    for i in range(1, refines + 1):
        tier = ((i - 1) // 20) + 1
        tier_counts[tier] += 1

    total = 0.0
    for tier in range(1, 6):
        total += tier_counts[tier] * tier_cost(tier)

    discounts = min(discount_refines, refines)

    # First discount always applies to Tier 1 if available
    if discounts > 0 and tier_counts[1] > 0:
        total -= tier_cost(1) * 0.5
        tier_counts[1] -= 1
        discounts -= 1

    # Remaining discounts apply to highest-cost tiers first
    for tier in range(5, 0, -1):
        if discounts == 0:
            break

        apply = min(discounts, tier_counts[tier])
        total -= apply * tier_cost(tier) * 0.5
        discounts -= apply

    return total


def cost_additional_this_week(
    additional_refines: int,
    used_this_week: int,
    *,
    discount_refines: int = 7,
) -> float:
    if additional_refines < 0:
        raise ValueError("additional_refines must be >= 0")
    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")
    if used_this_week + additional_refines > 100:
        raise ValueError("used_this_week + additional_refines cannot exceed 100")

    return (
        cost_one_week(used_this_week + additional_refines, discount_refines=discount_refines)
        - cost_one_week(used_this_week, discount_refines=discount_refines)
    )


def total_cost_even_distribution(refines: int, weeks: int | None = None) -> float:
    """
    Match the VBA TG_Cost behavior.
    """

    if refines <= 0:
        return 0.0

    if weeks is None:
        weeks = (refines + 99) // 100

    if refines > 100 * weeks:
        raise ValueError("refines exceed weekly capacity")

    base = refines // weeks
    extra = refines % weeks

    total = 0.0

    for wk in range(1, weeks + 1):
        this_week = base + (1 if wk <= extra else 0)
        total += cost_one_week(this_week)

    return total