from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import polars as pl


BUILDINGS = (
    "TC",
    "Embassy",
    "Command Center",
    "Infirmary",
    "Barracks",
    "Range",
    "Stable",
    "War Academy",
)


def _normalize_building_name(name: str) -> str:
    return " ".join(str(name).strip().replace("_", " ").replace("-", " ").split()).lower()


_BUILDING_ALIASES = {
    "tc": "TC",
    "town center": "TC",
    "embassy": "Embassy",
    "command center": "Command Center",
    "cc": "Command Center",
    "infirmary": "Infirmary",
    "barracks": "Barracks",
    "range": "Range",
    "stable": "Stable",
    "war academy": "War Academy",
    "academy": "War Academy",
}

_NORMALIZED_BUILDING_MAP = {
    _normalize_building_name(alias): canonical
    for alias, canonical in _BUILDING_ALIASES.items()
}


def canonical_building_name(name: str) -> str:
    key = _normalize_building_name(name)
    if key not in _NORMALIZED_BUILDING_MAP:
        raise KeyError(f"Unknown building name: {name}")
    return _NORMALIZED_BUILDING_MAP[key]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_CSV = DATA_DIR / "scenarios.csv"
BUILDING_COSTS_CSV = DATA_DIR / "building_costs.csv"
TC_PREREQS_CSV = DATA_DIR / "tc_prereqs.csv"


@dataclass(frozen=True)
class LookupTotals:
    truegold: int
    tempered_truegold: int





@lru_cache(maxsize=1)
def load_scenarios() -> pl.DataFrame:
    if not SCENARIOS_CSV.exists():
        raise FileNotFoundError(f"Missing scenarios CSV: {SCENARIOS_CSV}")

    df = pl.read_csv(SCENARIOS_CSV)

    required = {"ScenarioID", "ScenarioName"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"scenarios.csv missing required columns: {sorted(missing)}")

    for building in BUILDINGS:
        if building not in df.columns:
            raise ValueError(f"scenarios.csv missing building column: {building}")

    return df



@lru_cache(maxsize=1)
def load_building_costs() -> pl.DataFrame:
    if not BUILDING_COSTS_CSV.exists():
        raise FileNotFoundError(f"Missing building costs CSV: {BUILDING_COSTS_CSV}")

    df = pl.read_csv(BUILDING_COSTS_CSV)

    field_map = {str(col).strip().lower(): col for col in df.columns}

    building_col = field_map.get("building")
    from_col = field_map.get("fromtg")
    to_col = field_map.get("totg")
    tg_col = field_map.get("tg", field_map.get("truegold"))
    ttg_col = field_map.get("ttg", field_map.get("temperedtruegold"))

    missing = []
    if building_col is None:
        missing.append("Building")
    if from_col is None:
        missing.append("FromTG")
    if to_col is None:
        missing.append("ToTG")
    if tg_col is None:
        missing.append("TG/Truegold")
    if ttg_col is None:
        missing.append("TTG/TemperedTruegold")

    if missing:
        raise ValueError(
            f"building_upgrade_costs.csv missing required columns: {missing}"
        )

    df = df.rename(
        {
            building_col: "Building",
            from_col: "FromTG",
            to_col: "ToTG",
            tg_col: "TG",
            ttg_col: "TTG",
        }
    )

    df = df.with_columns(
        pl.col("Building").map_elements(canonical_building_name, return_dtype=pl.String),
        pl.col("FromTG").cast(pl.Int64),
        pl.col("ToTG").cast(pl.Int64),
        pl.col("TG").cast(pl.Int64),
        pl.col("TTG").cast(pl.Int64),
    ).sort(["Building", "FromTG", "ToTG"])

    return df

@lru_cache(maxsize=1)
def load_tc_prereqs() -> pl.DataFrame:
    """
    Expected CSV columns:
      TargetTG, EmbassyReq, BarracksReq, RangeReq, StableReq

    Notes column is optional and ignored.
    """
    if not TC_PREREQS_CSV.exists():
        return pl.DataFrame(
            {
                "TargetTG": [],
                "EmbassyReq": [],
                "BarracksReq": [],
                "RangeReq": [],
                "StableReq": [],
            }
        )

    df = pl.read_csv(TC_PREREQS_CSV)

    field_map = {str(col).strip().lower(): col for col in df.columns}

    target_col = field_map.get("targettg")
    embassy_col = field_map.get("embassyreq")
    barracks_col = field_map.get("barracksreq")
    range_col = field_map.get("rangereq")
    stable_col = field_map.get("stablereq")

    missing = []
    if target_col is None:
        missing.append("TargetTG")
    if embassy_col is None:
        missing.append("EmbassyReq")
    if barracks_col is None:
        missing.append("BarracksReq")
    if range_col is None:
        missing.append("RangeReq")
    if stable_col is None:
        missing.append("StableReq")

    if missing:
        raise ValueError(f"tc_prereqs.csv missing required columns: {missing}")

    df = df.rename(
        {
            target_col: "TargetTG",
            embassy_col: "EmbassyReq",
            barracks_col: "BarracksReq",
            range_col: "RangeReq",
            stable_col: "StableReq",
        }
    )

    df = df.with_columns(
        pl.col("TargetTG").cast(pl.Int64),
        pl.col("EmbassyReq").cast(pl.Int64),
        pl.col("BarracksReq").cast(pl.Int64),
        pl.col("RangeReq").cast(pl.Int64),
        pl.col("StableReq").cast(pl.Int64),
    ).sort("TargetTG")

    return df


@lru_cache(maxsize=None)
def tc_requirements_for_target(target_tg: int) -> dict[str, int]:
    """
    Return the support-building requirements to reach a given TC target tier.

    Example output:
      {
        "Embassy": 6,
        "Barracks": 5,
        "Range": 6,
        "Stable": 4,
      }
    """
    target_tg = int(target_tg)

    df = load_tc_prereqs()
    if df.height == 0:
        return {}

    row = df.filter(pl.col("TargetTG") == target_tg)
    if row.height == 0:
        raise KeyError(f"No TC prerequisite row found for TargetTG={target_tg}")

    return {
        "Embassy": int(row[0, "EmbassyReq"]),
        "Barracks": int(row[0, "BarracksReq"]),
        "Range": int(row[0, "RangeReq"]),
        "Stable": int(row[0, "StableReq"]),
    }


def expand_required_targets_from_tc_prereqs(
    requested_targets: Mapping[str, int],
) -> dict[str, int]:
    """
    Expand requested building targets using tc_prereqs.csv.

    Logic:
    - keep requested targets
    - infer required TC target as the max requested TC level
    - if TC is required, add Embassy/Barracks/Range/Stable minimums from tc_prereqs.csv
    - repeat until stable in case the expanded targets imply a higher TC target later

    This is intentionally simple and matches the structure of tc_prereqs.csv.
    """
    required = {
        canonical_building_name(building): int(level)
        for building, level in requested_targets.items()
        if int(level) > 0
    }

    changed = True
    while changed:
        changed = False

        tc_target = int(required.get("TC", 0))
        if tc_target > 0:
            tc_reqs = tc_requirements_for_target(tc_target)

            for building, req_level in tc_reqs.items():
                old = int(required.get(building, 0))
                if int(req_level) > old:
                    required[building] = int(req_level)
                    changed = True

    return required


@lru_cache(maxsize=None)
def get_scenario_flags(scenario_id: str) -> dict[str, bool]:
    df = load_scenarios()
    row = df.filter(pl.col("ScenarioID") == scenario_id)

    if row.height == 0:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    return {building: bool(row[0, building]) for building in BUILDINGS}


@lru_cache(maxsize=None)
def get_scenario_name(scenario_id: str) -> str:
    df = load_scenarios()
    row = df.filter(pl.col("ScenarioID") == scenario_id)

    if row.height == 0:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    return str(row[0, "ScenarioName"])


def list_scenarios() -> list[tuple[str, str]]:
    df = load_scenarios()
    return list(df.select(["ScenarioID", "ScenarioName"]).iter_rows())


def _normalize_level_map(levels: Mapping[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, level in levels.items():
        out[canonical_building_name(name)] = int(level)
    return out


def fill_missing_levels(
    levels: Mapping[str, int],
    default_level: int,
) -> dict[str, int]:
    normalized = _normalize_level_map(levels)
    return {
        building: int(normalized.get(building, default_level))
        for building in BUILDINGS
    }


def uniform_levels(level: int) -> dict[str, int]:
    return {building: int(level) for building in BUILDINGS}


@lru_cache(maxsize=None)
def lookup_building_cost(building: str, start_tg: int, target_tg: int) -> LookupTotals:
    """
    Sum stepwise costs from start_tg to target_tg using rows like:
      Building, FromTG, ToTG, TG, TTG
    """
    building = canonical_building_name(building)
    start_tg = int(start_tg)
    target_tg = int(target_tg)

    if target_tg < start_tg:
        raise ValueError(
            f"target_tg must be >= start_tg for {building}: {start_tg} -> {target_tg}"
        )

    if target_tg == start_tg:
        return LookupTotals(truegold=0, tempered_truegold=0)

    df = load_building_costs()

    rows = df.filter(
        (pl.col("Building") == building)
        & (pl.col("FromTG") >= start_tg)
        & (pl.col("ToTG") <= target_tg)
    ).sort(["FromTG", "ToTG"])

    if rows.height == 0:
        raise KeyError(
            f"No stepwise rows found for building={building}, start_tg={start_tg}, target_tg={target_tg}"
        )

    expected_from = start_tg
    total_tg = 0
    total_ttg = 0

    for row in rows.iter_rows(named=True):
        from_tg = int(row["FromTG"])
        to_tg = int(row["ToTG"])

        if from_tg != expected_from:
            raise ValueError(
                f"Missing cost step for {building}: expected FromTG={expected_from}, found FromTG={from_tg}"
            )

        if to_tg != from_tg + 1:
            raise ValueError(
                f"Non-step row for {building}: {from_tg}->{to_tg}. Expected one-level step."
            )

        total_tg += int(row["TG"])
        total_ttg += int(row["TTG"])
        expected_from = to_tg

    if expected_from != target_tg:
        raise ValueError(
            f"Incomplete step chain for {building}: ended at TG {expected_from}, expected {target_tg}"
        )

    return LookupTotals(
        truegold=total_tg,
        tempered_truegold=total_ttg,
    )


def lookup_custom_totals(
    *,
    start_levels: Mapping[str, int],
    target_levels: Mapping[str, int],
    scenario_flags: Mapping[str, bool],
) -> LookupTotals:
    """
    Core aggregation engine using per-building levels + inclusion flags,
    with TC prerequisite expansion from tc_prereqs.csv.
    """
    start_levels = _normalize_level_map(start_levels)
    target_levels = _normalize_level_map(target_levels)
    flags = {
        canonical_building_name(k): bool(v)
        for k, v in scenario_flags.items()
    }

    # Step 1: requested targets from the scenario itself
    requested_targets: dict[str, int] = {}
    for building in BUILDINGS:
        if not flags.get(building, False):
            continue

        start = int(start_levels.get(building, 0))
        target = int(target_levels.get(building, start))

        if target < start:
            raise ValueError(
                f"Target level must be >= start level for {building}: {start} -> {target}"
            )

        requested_targets[building] = target

    # Step 2: expand using TC prerequisite table
    required_targets = expand_required_targets_from_tc_prereqs(requested_targets)

    # Step 3: sum costs from current/start to required target
    total_tg = 0
    total_ttg = 0

    for building, required_target in required_targets.items():
        start = int(start_levels.get(building, 0))

        if required_target < start:
            # already satisfied
            continue

        cost = lookup_building_cost(building, start, required_target)
        total_tg += cost.truegold
        total_ttg += cost.tempered_truegold

    return LookupTotals(
        truegold=int(total_tg),
        tempered_truegold=int(total_ttg),
    )


def lookup_scenario_totals(
    *,
    start_tg_level: int,
    target_tg_level: int,
    scenario_id: str,
) -> LookupTotals:
    if int(target_tg_level) < int(start_tg_level):
        raise ValueError("target_tg_level must be >= start_tg_level")

    return lookup_custom_totals(
        start_levels=uniform_levels(int(start_tg_level)),
        target_levels=uniform_levels(int(target_tg_level)),
        scenario_flags=get_scenario_flags(scenario_id),
    )


def lookup_scenario_totals_advanced(
    *,
    start_levels: Mapping[str, int],
    target_levels: Mapping[str, int],
    scenario_id: str,
    default_start_level: int = 5,
) -> LookupTotals:
    filled_start = fill_missing_levels(start_levels, default_start_level)
    filled_target = fill_missing_levels(target_levels, default_start_level)

    return lookup_custom_totals(
        start_levels=filled_start,
        target_levels=filled_target,
        scenario_flags=get_scenario_flags(scenario_id),
    )