from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from ttg.cost_lookup import lookup_target_ttg
from ttg.planner import (
    DEFAULT_MAX_FRONTIER_STATES,
    DEFAULT_PROB_BUCKET,
    all_in_plan,
    compare_plan_set,
    evaluate_plan,
    even_split_plan,
    normalize_probability_input,
    optimal_plan_from_lookup,
    plan_table_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_PROBABILITIES: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def _normalize_probability_list(
    probabilities: Iterable[float | int] | None,
) -> list[float]:
    if probabilities is None:
        return list(DEFAULT_PROBABILITIES)

    return [float(normalize_probability_input(p)) for p in probabilities]


def build_plan_table(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    probabilities: Iterable[float | int] | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
) -> pl.DataFrame:
    """
    Build a Polars table of optimal plans for one TG/scenario combination.
    """
    prob_list = _normalize_probability_list(probabilities)

    rows = plan_table_rows(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        scenario=scenario,
        weeks=weeks,
        probabilities=prob_list,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
        prob_bucket=prob_bucket,
        max_frontier_states=max_frontier_states,
    )

    df = pl.DataFrame(rows)

    preferred_order = [
        "start_tg_level",
        "target_tg_level",
        "scenario",
        "requested_probability",
        "target_ttg",
        "current_ttg",
        "used_this_week",
        "last_week_days",
    ]
    preferred_order += [f"week_{i}" for i in range(1, weeks + 1)]
    preferred_order += [
        "total_refines",
        "total_cost",
        "achieved_probability",
    ]

    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]

    return df.select(existing + remaining)


def build_multi_scenario_table(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenarios: Sequence[str],
    weeks: int,
    probabilities: Iterable[float | int] | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
) -> pl.DataFrame:
    """
    Build one combined table for multiple scenarios.
    """
    tables: list[pl.DataFrame] = []

    for scenario in scenarios:
        tables.append(
            build_plan_table(
                start_tg_level=start_tg_level,
                target_tg_level=target_tg_level,
                scenario=scenario,
                weeks=weeks,
                probabilities=probabilities,
                current_ttg=current_ttg,
                used_this_week=used_this_week,
                last_week_days=last_week_days,
                prob_bucket=prob_bucket,
                max_frontier_states=max_frontier_states,
            )
        )

    if not tables:
        return pl.DataFrame()

    return pl.concat(tables, how="vertical")


def build_standard_two_week_tables(
    *,
    start_tg_level: int,
    target_tg_level: int,
    probabilities: Iterable[float | int] | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
) -> dict[str, pl.DataFrame]:
    """
    Build the four standard 2-week scenario tables.
    """
    prob_list = _normalize_probability_list(probabilities)

    configs = {
        "troops_2w": "Troops",
        "troops_cc_2w": "Troops+CC",
        "all_buildings_2w": "All Buildings",
    }

    result: dict[str, pl.DataFrame] = {}

    for name, scenario in configs.items():
        result[name] = build_plan_table(
            start_tg_level=start_tg_level,
            target_tg_level=target_tg_level,
            scenario=scenario,
            weeks=2,
            probabilities=prob_list,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
            prob_bucket=prob_bucket,
            max_frontier_states=max_frontier_states,
        )

    return result


def compare_plan_variants(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    desired_probability: float | int | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    include_all_in: bool = True,
    include_even: bool = True,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
) -> pl.DataFrame:
    """
    Compare optimizer vs simple benchmark plans for the same scenario.

    Rows included:
    - optimized
    - even_split_same_total
    - all_in
    """
    result = optimal_plan_from_lookup(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        scenario=scenario,
        weeks=weeks,
        desired_probability=desired_probability,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
        prob_bucket=prob_bucket,
        max_frontier_states=max_frontier_states,
    )

    target_ttg = result.target_ttg
    requested_probability = normalize_probability_input(desired_probability)

    rows: list[dict] = []

    opt_row = {
        "plan_type": "optimized",
        "start_tg_level": start_tg_level,
        "target_tg_level": target_tg_level,
        "scenario": scenario,
        "requested_probability": requested_probability,
        "target_ttg": target_ttg,
        "current_ttg": current_ttg,
        "used_this_week": used_this_week,
        "last_week_days": last_week_days,
        "total_refines": result.total_refines,
        "total_cost": result.total_cost,
        "achieved_probability": result.achieved_probability,
    }
    for i, refs in enumerate(result.week_refines, start=1):
        opt_row[f"week_{i}"] = refs
    rows.append(opt_row)

    if include_even:
        even_plan = even_split_plan(
            result.total_refines,
            weeks,
            used_this_week,
            front_load_remainder=True,
        )
        even_eval = evaluate_plan(
            target_ttg=target_ttg,
            week_refines=even_plan,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )

        even_row = {
            "plan_type": "even_split_same_total",
            "start_tg_level": start_tg_level,
            "target_tg_level": target_tg_level,
            "scenario": scenario,
            "requested_probability": requested_probability,
            "target_ttg": target_ttg,
            "current_ttg": current_ttg,
            "used_this_week": used_this_week,
            "last_week_days": last_week_days,
            "total_refines": even_eval["total_refines"],
            "total_cost": even_eval["total_cost"],
            "achieved_probability": even_eval["achieved_probability"],
        }
        for i, refs in enumerate(even_plan, start=1):
            even_row[f"week_{i}"] = refs
        rows.append(even_row)

    if include_all_in:
        all_in = all_in_plan(weeks, used_this_week)
        all_in_eval = evaluate_plan(
            target_ttg=target_ttg,
            week_refines=all_in,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )

        all_in_row = {
            "plan_type": "all_in",
            "start_tg_level": start_tg_level,
            "target_tg_level": target_tg_level,
            "scenario": scenario,
            "requested_probability": requested_probability,
            "target_ttg": target_ttg,
            "current_ttg": current_ttg,
            "used_this_week": used_this_week,
            "last_week_days": last_week_days,
            "total_refines": all_in_eval["total_refines"],
            "total_cost": all_in_eval["total_cost"],
            "achieved_probability": all_in_eval["achieved_probability"],
        }
        for i, refs in enumerate(all_in, start=1):
            all_in_row[f"week_{i}"] = refs
        rows.append(all_in_row)

    df = pl.DataFrame(rows)

    preferred_order = [
        "plan_type",
        "start_tg_level",
        "target_tg_level",
        "scenario",
        "requested_probability",
        "target_ttg",
        "current_ttg",
        "used_this_week",
        "last_week_days",
    ]
    preferred_order += [f"week_{i}" for i in range(1, weeks + 1)]
    preferred_order += ["total_refines", "total_cost", "achieved_probability"]

    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]

    return df.select(existing + remaining)


def compare_custom_plan_to_lookup(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    week_refines: Iterable[int],
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
) -> pl.DataFrame:
    """
    Evaluate one custom plan against the lookup target.
    """
    plan_tuple = tuple(int(x) for x in week_refines)
    target_ttg = lookup_target_ttg(start_tg_level, target_tg_level, scenario)
    result = evaluate_plan(
        target_ttg=target_ttg,
        week_refines=plan_tuple,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )

    row = {
        "plan_type": "custom",
        "start_tg_level": start_tg_level,
        "target_tg_level": target_tg_level,
        "scenario": scenario,
        "target_ttg": target_ttg,
        "current_ttg": current_ttg,
        "used_this_week": used_this_week,
        "last_week_days": last_week_days,
        "total_refines": result["total_refines"],
        "total_cost": result["total_cost"],
        "achieved_probability": result["achieved_probability"],
    }

    for i, refs in enumerate(plan_tuple, start=1):
        row[f"week_{i}"] = refs

    return pl.DataFrame([row])


def compare_optimizer_vs_custom(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    custom_plan: Iterable[int],
    desired_probability: float | int | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
) -> pl.DataFrame:
    """
    Compare optimizer result directly against a user-provided custom plan.
    """
    opt = compare_plan_variants(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        scenario=scenario,
        weeks=weeks,
        desired_probability=desired_probability,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
        include_all_in=False,
        include_even=False,
        prob_bucket=prob_bucket,
        max_frontier_states=max_frontier_states,
    )

    custom = compare_custom_plan_to_lookup(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        scenario=scenario,
        week_refines=custom_plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )

    return pl.concat([opt, custom], how="vertical")


def export_table_csv(df: pl.DataFrame, filename: str, *, output_dir: Path | None = None) -> Path:
    """
    Export a single Polars DataFrame to CSV.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    df.write_csv(path)
    return path


def export_standard_two_week_tables_csv(
    *,
    start_tg_level: int,
    target_tg_level: int,
    probabilities: Iterable[float | int] | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Build and export the standard 2-week tables to CSV files.
    """
    tables = build_standard_two_week_tables(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        probabilities=probabilities,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
        prob_bucket=prob_bucket,
        max_frontier_states=max_frontier_states,
    )

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.write_csv(path)
        paths[name] = path

    return paths