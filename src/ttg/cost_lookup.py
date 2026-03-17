from __future__ import annotations

from typing import Mapping

from ttg.newcostlookup import (
    BUILDINGS,
    LookupTotals,
    fill_missing_levels,
    get_scenario_flags,
    get_scenario_name,
    list_scenarios,
    lookup_custom_totals,
    lookup_scenario_totals,
    lookup_scenario_totals_advanced,
)


def lookup_targets(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
) -> LookupTotals:
    """
    Backward-compatible simple lookup API.

    Applies one start TG level and one target TG level to all buildings,
    then filters by scenario membership.
    """
    return lookup_scenario_totals(
        start_tg_level=int(start_tg_level),
        target_tg_level=int(target_tg_level),
        scenario_id=str(scenario),
    )


def lookup_target_ttg(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
) -> int:
    """
    Convenience helper returning TTG only.
    """
    return int(
        lookup_targets(
            start_tg_level=start_tg_level,
            target_tg_level=target_tg_level,
            scenario=scenario,
        ).tempered_truegold
    )


def lookup_target_tg(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
) -> int:
    """
    Convenience helper returning TG only.
    """
    return int(
        lookup_targets(
            start_tg_level=start_tg_level,
            target_tg_level=target_tg_level,
            scenario=scenario,
        ).truegold
    )


def lookup_targets_advanced(
    *,
    start_levels: Mapping[str, int],
    target_levels: Mapping[str, int],
    scenario: str | None = None,
    scenario_flags: Mapping[str, bool] | None = None,
    default_start_level: int = 5,
) -> LookupTotals:
    """
    Advanced lookup using per-building start and target TG levels.

    You must provide either:
      - scenario
      - scenario_flags

    If scenario is provided, its building membership comes from scenarios.csv.
    If scenario_flags is provided, it is used directly.
    """
    if scenario is None and scenario_flags is None:
        raise ValueError("Provide either scenario or scenario_flags")

    filled_start = fill_missing_levels(start_levels, default_start_level)
    filled_target = fill_missing_levels(target_levels, default_start_level)

    for building in BUILDINGS:
        if int(filled_target[building]) < int(filled_start[building]):
            raise ValueError(
                f"Target level must be >= start level for {building}: "
                f"{filled_start[building]} -> {filled_target[building]}"
            )

    if scenario_flags is None:
        return lookup_scenario_totals_advanced(
            start_levels=filled_start,
            target_levels=filled_target,
            scenario_id=str(scenario),
            default_start_level=default_start_level,
        )

    normalized_flags = {
        building: bool(scenario_flags.get(building, False))
        for building in BUILDINGS
    }

    return lookup_custom_totals(
        start_levels=filled_start,
        target_levels=filled_target,
        scenario_flags=normalized_flags,
    )


def lookup_target_ttg_advanced(
    *,
    start_levels: Mapping[str, int],
    target_levels: Mapping[str, int],
    scenario: str | None = None,
    scenario_flags: Mapping[str, bool] | None = None,
    default_start_level: int = 5,
) -> int:
    """
    Advanced TTG-only helper.
    """
    return int(
        lookup_targets_advanced(
            start_levels=start_levels,
            target_levels=target_levels,
            scenario=scenario,
            scenario_flags=scenario_flags,
            default_start_level=default_start_level,
        ).tempered_truegold
    )


def lookup_target_tg_advanced(
    *,
    start_levels: Mapping[str, int],
    target_levels: Mapping[str, int],
    scenario: str | None = None,
    scenario_flags: Mapping[str, bool] | None = None,
    default_start_level: int = 5,
) -> int:
    """
    Advanced TG-only helper.
    """
    return int(
        lookup_targets_advanced(
            start_levels=start_levels,
            target_levels=target_levels,
            scenario=scenario,
            scenario_flags=scenario_flags,
            default_start_level=default_start_level,
        ).truegold
    )


__all__ = [
    "BUILDINGS",
    "LookupTotals",
    "get_scenario_flags",
    "get_scenario_name",
    "list_scenarios",
    "lookup_targets",
    "lookup_target_tg",
    "lookup_target_ttg",
    "lookup_targets_advanced",
    "lookup_target_tg_advanced",
    "lookup_target_ttg_advanced",
]