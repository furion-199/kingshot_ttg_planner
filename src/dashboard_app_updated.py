from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
import streamlit as st

from ttg.cost_lookup import lookup_targets, lookup_targets_advanced
from ttg.newcostlookup import expand_required_targets_from_tc_prereqs, lookup_building_cost, get_scenario_flags
from ttg.planner import (
    compare_plan_set,
    evaluate_plan,
    inspect_candidate_family,
    optimal_plan,
    optimal_plan_from_lookup,
)
from ttg.tables import build_plan_table

st.set_page_config(page_title="Kingshot TTG Planner", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 360px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 360px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

TG_LEVELS = list(range(0, 9))
DEFAULT_PROBS = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BUILDINGS = ["TC", "Embassy", "Command Center", "Infirmary", "Barracks", "Range", "Stable", "War Academy"]


@st.cache_data(show_spinner=False)
def _load_scenarios() -> list[tuple[str, str]]:
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    csv_path = project_root / "data" / "scenarios.csv"

    fallback = [
        ("TC_RBS", "TC + all 3 troop buildings"),
        ("TC_R", "TC + Range"),
        ("TC_B", "TC + Barracks"),
        ("TC_S", "TC + Stable"),
        ("TC_RB", "TC + Range + Barracks"),
        ("TC_RS", "TC + Range + Stable"),
        ("TC_BS", "TC + Barracks + Stable"),
        ("TC_CC", "TC + Command Center"),
        ("TC_RBS_CC", "all 3 troop buildings + CC"),
        ("ALL_TG", "buildings + War Academy"),
    ]

    if not csv_path.exists():
        return fallback

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return fallback

            field_map = {name.strip().lower(): name for name in reader.fieldnames}
            id_col = field_map.get("scenarioid") or field_map.get("id")
            name_col = field_map.get("scenarioname") or field_map.get("name")

            if id_col is None or name_col is None:
                return fallback

            rows: list[tuple[str, str]] = []
            for row in reader:
                sid = (row.get(id_col) or "").strip()
                sname = (row.get(name_col) or "").strip()
                if sid:
                    rows.append((sid, sname if sname else sid))

            if rows:
                seen = set()
                deduped = []
                for item in rows:
                    if item[0] not in seen:
                        seen.add(item[0])
                        deduped.append(item)
                return deduped
    except Exception:
        return fallback

    return fallback


SCENARIO_OPTIONS = _load_scenarios()
SCENARIO_KEYS = [k for k, _ in SCENARIO_OPTIONS]
SCENARIO_LABELS = {k: v for k, v in SCENARIO_OPTIONS}


@st.cache_data(show_spinner=False)
def _lookup_totals_simple(start_tg_level: int, target_tg_level: int, scenario: str) -> dict:
    totals = lookup_targets(start_tg_level, target_tg_level, scenario)
    return {"target_tg": totals.truegold, "target_ttg": totals.tempered_truegold}


@st.cache_data(show_spinner=False)
def _lookup_totals_advanced(
    start_levels: tuple[tuple[str, int], ...],
    target_levels: tuple[tuple[str, int], ...],
    default_start_level: int,
) -> dict:
    start_dict = dict(start_levels)
    target_dict = dict(target_levels)

    scenario_flags = {
        building: int(target_dict.get(building, default_start_level)) > int(start_dict.get(building, default_start_level))
        for building in BUILDINGS
    }

    totals = lookup_targets_advanced(
        start_levels=start_dict,
        target_levels=target_dict,
        scenario_flags=scenario_flags,
        default_start_level=default_start_level,
    )
    return {"target_tg": totals.truegold, "target_ttg": totals.tempered_truegold}


@st.cache_data(show_spinner=False)
def _build_plan_table_cached(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    probabilities: tuple[float, ...],
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
) -> pl.DataFrame:
    return build_plan_table(
        start_tg_level=start_tg_level,
        target_tg_level=target_tg_level,
        scenario=scenario,
        weeks=weeks,
        probabilities=list(probabilities),
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )


@st.cache_data(show_spinner=False)
def _optimal_plan_simple_cached(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    desired_probability: float | None,
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
):
    try:
        return optimal_plan_from_lookup(
            start_tg_level=start_tg_level,
            target_tg_level=target_tg_level,
            scenario=scenario,
            weeks=weeks,
            desired_probability=desired_probability,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _optimal_plan_advanced_cached(
    target_ttg: int,
    weeks: int,
    desired_probability: float | None,
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
):
    try:
        return optimal_plan(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=desired_probability,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _compare_plan_set_cached(
    target_ttg: int,
    custom_plan: tuple[int, ...],
    weeks: int,
    desired_probability: float | None,
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
):
    rows = compare_plan_set(
        target_ttg=target_ttg,
        custom_plan=custom_plan,
        weeks=weeks,
        desired_probability=desired_probability,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )
    return pl.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _candidate_family_cached(
    target_ttg: int,
    total_refines: int,
    weeks: int,
    current_ttg: int,
    used_this_week: int,
) -> pl.DataFrame:
    rows = inspect_candidate_family(
        target_ttg,
        total_refines=total_refines,
        weeks=weeks,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
    )
    return pl.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _evaluate_custom_plan_cached(
    target_ttg: int,
    custom_plan: tuple[int, ...],
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
) -> dict:
    return evaluate_plan(
        target_ttg=target_ttg,
        week_refines=custom_plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )


@st.cache_data(show_spinner=False)
def _build_advanced_probability_table(
    target_ttg: int,
    building_tg_cost: int,
    weeks: int,
    probabilities: tuple[float, ...],
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
) -> pl.DataFrame:
    rows = []
    for p in probabilities:
        result = _optimal_plan_advanced_cached(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=p,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )
        row = {
            "requested_probability": p,
            "target_ttg": target_ttg,
            "current_ttg": current_ttg,
            "used_this_week": used_this_week,
            "last_week_days": last_week_days,
            "building_tg_cost": building_tg_cost,
        }
        if result is None:
            for i in range(1, weeks + 1):
                row[f"week_{i}"] = None
            row["total_refines"] = None
            row["refinement_tg_cost"] = None
            row["total_tg_cost"] = None
            row["achieved_probability"] = None
            row["status"] = "Unachievable"
        else:
            for i, refs in enumerate(result.week_refines, start=1):
                row[f"week_{i}"] = refs
            row["total_refines"] = result.total_refines
            row["refinement_tg_cost"] = result.total_cost
            row["total_tg_cost"] = building_tg_cost + result.total_cost
            row["achieved_probability"] = result.achieved_probability
            row["status"] = "OK"
        rows.append(row)
    return pl.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _simple_scenario_summary(start_tg_level: int, target_tg_level: int, scenario: str) -> pd.DataFrame:
    scenario_flags = get_scenario_flags(scenario)
    requested_targets = {
        b: int(target_tg_level)
        for b in BUILDINGS
        if bool(scenario_flags.get(b, False))
    }
    expanded_targets = expand_required_targets_from_tc_prereqs(requested_targets)

    rows = []
    for b in BUILDINGS:
        start = int(start_tg_level)
        requested = requested_targets.get(b)
        expanded = expanded_targets.get(b)
        forced = expanded is not None and (requested is None or int(expanded) > int(requested))

        tg_cost = None
        ttg_cost = None
        final_target = None
        if expanded is not None and int(expanded) > start:
            final_target = int(expanded)
            try:
                cost = lookup_building_cost(b, start, final_target)
                tg_cost = cost.truegold
                ttg_cost = cost.tempered_truegold
            except Exception:
                tg_cost = None
                ttg_cost = None

        rows.append(
            {
                "building": b,
                "start": start,
                "requested_target": int(requested) if requested is not None else None,
                "final_target": final_target,
                "forced_by_prereq": forced,
                "tg_cost": tg_cost,
                "ttg_cost": ttg_cost,
                "included_in_scenario": bool(scenario_flags.get(b, False)),
                "scenario": scenario,
            }
        )

    return pd.DataFrame(rows)


def _normalize_probability_text(value: str) -> float | None:
    value = value.strip()
    if value == "":
        return None
    p = float(value)
    if p > 1:
        p = p / 100
    if p <= 0 or p > 1:
        raise ValueError("Probability must be in (0,1] or (0,100].")
    return p


def _parse_plan(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return tuple()
    try:
        if text.startswith("["):
            vals = ast.literal_eval(text)
            if not isinstance(vals, (list, tuple)):
                raise ValueError
            return tuple(int(x) for x in vals)
        return tuple(int(part.strip()) for part in text.split(",") if part.strip() != "")
    except Exception as exc:
        raise ValueError("Use comma-separated integers like 38,38,38 or a Python-style list.") from exc


def _display_metrics(result, building_tg_cost: int, target_ttg: int) -> None:
    cols = st.columns(5)
    cols[0].metric("Target TTG", f"{target_ttg:,}")
    cols[1].metric("Building Cost (TG)", f"{building_tg_cost:,}")
    if result is None:
        cols[2].metric("Refinement Cost (TG)", "—")
        cols[3].metric("Total TG", "—")
        cols[4].metric("Achieved Probability", "—")
        return
    cols[2].metric("Refinement Cost (TG)", f"{result.total_cost:,.0f}")
    cols[3].metric("Total TG", f"{building_tg_cost + result.total_cost:,.0f}")
    cols[4].metric("Achieved Probability", f"{result.achieved_probability:.4%}")


def _week_columns(plan: Iterable[int]) -> dict[str, int]:
    return {f"Week {i}": int(v) for i, v in enumerate(plan, start=1)}


def _plan_bar_df(plan: Iterable[int]) -> pd.DataFrame:
    vals = list(plan)
    return pd.DataFrame({"Week": [f"Week {i}" for i in range(1, len(vals) + 1)], "Refines": vals}).set_index("Week")


def _daily_schedule_for_plan(plan: Iterable[int], *, last_week_days: int = 5) -> pd.DataFrame:
    vals = list(int(x) for x in plan)
    weeks = len(vals)
    data: dict[str, list[int]] = {}
    for week_idx, refs in enumerate(vals, start=1):
        days_this_week = last_week_days if week_idx == weeks else 7
        schedule = [0] * days_this_week
        if refs > 0:
            active_days = min(refs, days_this_week)
            if refs <= active_days:
                for i in range(active_days):
                    schedule[i] = 1
            else:
                schedule[0] = refs - (active_days - 1)
                for i in range(1, active_days):
                    schedule[i] = 1
        full_col = [0] * 7
        for i, v in enumerate(schedule):
            full_col[i] = v
        data[f"Week {week_idx}"] = full_col
    return pd.DataFrame(data, index=DAY_NAMES)


def _levels_to_tuple(level_map: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((b, int(level_map[b])) for b in BUILDINGS)


def _augment_cost_columns(df: pd.DataFrame, building_tg_cost: int) -> pd.DataFrame:
    out = df.copy()
    if "total_cost" in out.columns and "refinement_tg_cost" not in out.columns:
        out = out.rename(columns={"total_cost": "refinement_tg_cost"})
    out["building_tg_cost"] = building_tg_cost
    if "refinement_tg_cost" in out.columns:
        out["total_tg_cost"] = out["building_tg_cost"] + out["refinement_tg_cost"].fillna(0)
        out.loc[out["refinement_tg_cost"].isna(), "total_tg_cost"] = None
    return out


def _download_csv_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv", key=key)


def _summary_df(mode_label: str, building_tg_cost: int, target_ttg: int, weeks: int, desired_probability: float | None, result) -> pd.DataFrame:
    row = {
        "mode": mode_label,
        "target_ttg": target_ttg,
        "building_tg_cost": building_tg_cost,
        "requested_probability": desired_probability,
        "weeks": weeks,
    }
    if result is None:
        row["total_refines"] = None
        row["refinement_tg_cost"] = None
        row["total_tg_cost"] = None
        row["achieved_probability"] = None
        row["status"] = "Unachievable"
    else:
        row["total_refines"] = result.total_refines
        row["refinement_tg_cost"] = result.total_cost
        row["total_tg_cost"] = building_tg_cost + result.total_cost
        row["achieved_probability"] = result.achieved_probability
        row["status"] = "OK"
        for i, refs in enumerate(result.week_refines, start=1):
            row[f"week_{i}"] = refs
    return pd.DataFrame([row])


def _validation_panel_advanced(building_inputs: dict[str, dict[str, int]], building_tg_cost: int, target_ttg: int) -> pd.DataFrame:
    requested_targets = {
        b: int(vals["target"])
        for b, vals in building_inputs.items()
        if int(vals["target"]) > int(vals["start"])
    }
    expanded_targets = expand_required_targets_from_tc_prereqs(requested_targets)

    rows = []
    for b in BUILDINGS:
        start = int(building_inputs[b]["start"])
        requested = int(requested_targets.get(b, start))
        expanded = int(expanded_targets.get(b, start))
        forced = expanded > requested

        tg_cost = None
        ttg_cost = None
        if expanded > start:
            try:
                cost = lookup_building_cost(b, start, expanded)
                tg_cost = cost.truegold
                ttg_cost = cost.tempered_truegold
            except Exception:
                tg_cost = None
                ttg_cost = None

        rows.append(
            {
                "building": b,
                "start": start,
                "requested_target": requested if requested > start else None,
                "final_target": expanded if expanded > start else None,
                "forced_by_prereq": forced,
                "tg_cost": tg_cost,
                "ttg_cost": ttg_cost,
            }
        )

    df = pd.DataFrame(rows)
    df["building_cost_tg_total"] = building_tg_cost
    df["target_ttg_total"] = target_ttg
    return df


st.title("Kingshot TTG Planner")
st.caption("Interactive dashboard for upgrade targets, optimal refine plans, and custom plan comparison.")

st.session_state.setdefault("simple_start_tg", 5)
st.session_state.setdefault("simple_target_tg", 7)
st.session_state.setdefault("simple_scenario", SCENARIO_KEYS[0] if SCENARIO_KEYS else "")
if "advanced_levels_df" not in st.session_state:
    st.session_state["advanced_levels_df"] = pd.DataFrame(
        {
            "Building": BUILDINGS,
            "Start": [5] * len(BUILDINGS),
            "Target": [7] * len(BUILDINGS),
        }
    )

with st.sidebar:
    st.header("Mode")
    advanced_mode = st.toggle("Advanced building-level inputs", value=False)

    st.header("Planning Inputs")
    weeks = st.slider("Weeks", min_value=1, max_value=16, value=2)
    current_ttg = st.number_input("Current TTG", min_value=0, value=0, step=1)
    used_this_week = st.number_input("Refines already used this week", min_value=0, max_value=100, value=0, step=1)
    last_week_days = st.slider("Last week active days", min_value=1, max_value=7, value=5, help="Event usually ends Friday → 5 discounted refines instead of 7")
    desired_probability_text = st.text_input("Desired probability", value="90%")

    try:
        desired_probability = _normalize_probability_text(desired_probability_text.replace("%", ""))
        prob_error = None
    except Exception as exc:
        desired_probability = None
        prob_error = str(exc)
    if prob_error:
        st.error(prob_error)

    target_tg = 0
    target_ttg = 0
    planning_mode = None
    building_inputs = None
    start_tg_level = None
    target_tg_level = None
    scenario = None

    if advanced_mode:
        st.header("Advanced Building Levels")
        c1, c2 = st.columns(2)
        adv_all_start = c1.selectbox("Set all starts", TG_LEVELS, index=TG_LEVELS.index(5), key="adv_all_start")
        adv_all_target = c2.selectbox("Set all targets", TG_LEVELS, index=TG_LEVELS.index(7), key="adv_all_target")

        c3, c4 = st.columns(2)
        if c3.button("Apply to all"):
            for b in BUILDINGS:
                st.session_state[f"adv_start_{b}"] = adv_all_start
                st.session_state[f"adv_target_{b}"] = adv_all_target
        if c4.button("Copy simple"):
            for b in BUILDINGS:
                st.session_state[f"adv_start_{b}"] = st.session_state["simple_start_tg"]
                st.session_state[f"adv_target_{b}"] = st.session_state["simple_target_tg"]
        if st.button("Reset advanced to TG5 → TG7"):
            for b in BUILDINGS:
                st.session_state[f"adv_start_{b}"] = 5
                st.session_state[f"adv_target_{b}"] = 7

        # initialize per-building widget state
        for b in BUILDINGS:
            st.session_state.setdefault(f"adv_start_{b}", 5)
            st.session_state.setdefault(f"adv_target_{b}", 7)

        st.caption("Tip: use Tab to move through fields. Each row is a normal widget pair for faster keyboard entry.")

        header_cols = st.columns([1.6, 1, 1, 0.8])
        header_cols[0].markdown("**Building**")
        header_cols[1].markdown("**Start**")
        header_cols[2].markdown("**Target**")
        header_cols[3].markdown("**Included**")

        building_inputs = {}
        advanced_valid = True
        summary_rows = []

        for b in BUILDINGS:
            row_cols = st.columns([1.6, 1, 1, 0.8])
            row_cols[0].write(b)
            start_b = row_cols[1].number_input(
                label=f"{b} start",
                min_value=min(TG_LEVELS),
                max_value=max(TG_LEVELS),
                step=1,
                key=f"adv_start_{b}",
                label_visibility="collapsed",
            )
            target_b = row_cols[2].number_input(
                label=f"{b} target",
                min_value=min(TG_LEVELS),
                max_value=max(TG_LEVELS),
                step=1,
                key=f"adv_target_{b}",
                label_visibility="collapsed",
            )
            included = int(target_b) > int(start_b)
            row_cols[3].write("Yes" if included else "No")

            if int(target_b) < int(start_b):
                advanced_valid = False
            building_inputs[b] = {"start": int(start_b), "target": int(target_b)}
            summary_rows.append({"Building": b, "Start": int(start_b), "Target": int(target_b), "Included": included})

        with st.expander("Advanced input summary", expanded=False):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if not advanced_valid:
            st.error("Each building target level must be at least its start level.")
        else:
            try:
                start_levels = _levels_to_tuple({b: vals["start"] for b, vals in building_inputs.items()})
                target_levels = _levels_to_tuple({b: vals["target"] for b, vals in building_inputs.items()})
                totals = _lookup_totals_advanced(start_levels=start_levels, target_levels=target_levels, default_start_level=0)
                target_tg = totals["target_tg"]
                target_ttg = totals["target_ttg"]
                planning_mode = "advanced"
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.write(f"**Building Cost (TG):** {target_tg:,}")
        st.write(f"**Target TTG:** {target_ttg:,}")
        st.caption("Advanced mode ignores scenarios and includes any building with target > start.")

    else:
        st.header("Simple Scenario Inputs")
        start_tg_level = st.selectbox("Start TG level", TG_LEVELS, key="simple_start_tg")
        target_tg_level = st.selectbox("Target TG level", TG_LEVELS, key="simple_target_tg")
        scenario = st.selectbox("Scenario", SCENARIO_KEYS, key="simple_scenario", format_func=lambda x: SCENARIO_LABELS.get(x, x))

        invalid_tg_range = target_tg_level <= start_tg_level
        if invalid_tg_range:
            st.error("Target TG level must be above start TG level.")
        else:
            try:
                totals = _lookup_totals_simple(start_tg_level, target_tg_level, scenario)
                target_tg = totals["target_tg"]
                target_ttg = totals["target_ttg"]
                planning_mode = "simple"
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.write(f"**Building Cost (TG):** {target_tg:,}")
        st.write(f"**Target TTG:** {target_ttg:,}")


simple_valid = planning_mode == "simple"


def _get_result_for_current_mode(probability: float | None):
    if planning_mode == "advanced":
        return _optimal_plan_advanced_cached(target_ttg=target_ttg, weeks=weeks, desired_probability=probability, current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
    if planning_mode == "simple":
        return _optimal_plan_simple_cached(start_tg_level=start_tg_level, target_tg_level=target_tg_level, scenario=scenario, weeks=weeks, desired_probability=probability, current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
    return None


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Optimal Plan", "Probability Table", "Compare Custom Plan", "Candidate Family", "Multi-Scenario View", "Validation"])

with tab1:
    st.subheader("Optimal Plan")
    result = _get_result_for_current_mode(desired_probability)
    _display_metrics(result, target_tg, target_ttg)

    if planning_mode is None:
        st.warning("Enter a valid target configuration.")
    elif planning_mode == "advanced":
        st.caption("Mode: Advanced building-level target lookup")
    else:
        st.caption("Mode: Simple scenario lookup")

    summary_pd = _summary_df(planning_mode or "none", target_tg, target_ttg, weeks, desired_probability, result)
    st.dataframe(summary_pd, use_container_width=True, hide_index=True)
    _download_csv_button(summary_pd, "Download summary card CSV", "summary_card.csv", "dl_summary_card")

    if planning_mode is not None and result is None:
        st.warning("Target probability cannot be achieved within the available weeks.")
    elif result is not None:
        week_df = pl.DataFrame([_week_columns(result.week_refines)])
        week_pd = week_df.to_pandas()
        st.dataframe(week_pd, use_container_width=True, hide_index=True)
        _download_csv_button(week_pd, "Download optimal plan CSV", "optimal_plan.csv", "dl_optimal_plan")
        st.markdown("### Weekly Refines")
        st.bar_chart(_plan_bar_df(result.week_refines))
        st.markdown("### Daily Refine Schedule")
        st.dataframe(_daily_schedule_for_plan(result.week_refines, last_week_days=last_week_days), use_container_width=True)

with tab2:
    st.subheader("Probability Table")
    if planning_mode is None:
        st.info("Enter a valid target configuration to view the probability table.")
    elif planning_mode == "advanced":
        st.caption("Mode: Advanced building-level target lookup")
        table_df = _build_advanced_probability_table(target_ttg=target_ttg, building_tg_cost=target_tg, weeks=weeks, probabilities=tuple(DEFAULT_PROBS), current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
        table_pd = table_df.to_pandas()
        st.dataframe(table_pd, use_container_width=True, hide_index=True)
        _download_csv_button(table_pd, "Download probability table CSV", "probability_table_advanced.csv", "dl_prob_adv")
    else:
        st.caption("Mode: Simple scenario lookup")
        table_df = _build_plan_table_cached(start_tg_level=start_tg_level, target_tg_level=target_tg_level, scenario=scenario, weeks=weeks, probabilities=tuple(DEFAULT_PROBS), current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
        table_pd = _augment_cost_columns(table_df.to_pandas(), target_tg)
        st.dataframe(table_pd, use_container_width=True, hide_index=True)
        _download_csv_button(table_pd, "Download probability table CSV", "probability_table_simple.csv", "dl_prob_simple")

with tab3:
    st.subheader("Compare Custom Plan")
    custom_plan_text = st.text_input("Custom weekly refines", value=",".join(["38"] * min(weeks, 8)) if weeks == 8 else ",".join(["0"] * weeks))
    try:
        custom_plan = _parse_plan(custom_plan_text)
        if len(custom_plan) != weeks:
            st.warning(f"Plan has {len(custom_plan)} weeks but current horizon is {weeks} weeks.")
        elif planning_mode is None:
            st.info("Enter a valid target configuration to compare plans.")
        else:
            cmp_df = _compare_plan_set_cached(target_ttg=target_ttg, custom_plan=custom_plan, weeks=weeks, desired_probability=desired_probability, current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
            cmp_pd = _augment_cost_columns(cmp_df.to_pandas(), target_tg)
            st.dataframe(cmp_pd, use_container_width=True, hide_index=True)
            _download_csv_button(cmp_pd, "Download comparison CSV", "compare_custom_plan.csv", "dl_compare_custom")

            custom_eval = _evaluate_custom_plan_cached(target_ttg=target_ttg, custom_plan=custom_plan, current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
            metric_cols = st.columns(3)
            metric_cols[0].metric("Refinement Cost (TG)", f"{custom_eval['total_cost']:,.0f}")
            metric_cols[1].metric("Building Cost (TG)", f"{target_tg:,.0f}")
            metric_cols[2].metric("Total TG", f"{target_tg + custom_eval['total_cost']:,.0f}")

            compare_choice = st.selectbox("Show schedule for", ["Custom Plan", "Optimizer"], index=0)
            schedule_plan = custom_plan
            if compare_choice == "Optimizer":
                opt_rows = cmp_df.filter(pl.col("plan_type") == "optimizer")
                if opt_rows.height > 0:
                    schedule_plan = tuple(int(opt_rows[0, "week_refines"][i]) for i in range(len(opt_rows[0, "week_refines"]))) if "week_refines" in opt_rows.columns else tuple(int(opt_rows[0, f"week_{i}"]) for i in range(1, weeks + 1))

            st.markdown("### Selected Plan Weekly Refines")
            st.bar_chart(_plan_bar_df(schedule_plan))
            st.markdown("### Selected Plan Daily Refine Schedule")
            st.dataframe(_daily_schedule_for_plan(schedule_plan, last_week_days=last_week_days), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

with tab4:
    st.subheader("Candidate Family Inspector")
    candidate_total = st.number_input("Total refines to inspect", min_value=0, max_value=int((100 - used_this_week) + 100 * (weeks - 1)), value=min(304, (100 - used_this_week) + 100 * (weeks - 1)), step=1)
    if planning_mode is None:
        st.info("Enter a valid target configuration to inspect candidates.")
    else:
        cand_df = _candidate_family_cached(target_ttg=target_ttg, total_refines=int(candidate_total), weeks=weeks, current_ttg=current_ttg, used_this_week=used_this_week)
        cand_pd = _augment_cost_columns(cand_df.to_pandas(), target_tg)
        sort_col = st.selectbox("Sort by", ["refinement_tg_cost", "achieved_probability", "total_refines", "total_tg_cost"], index=0)
        descending = st.checkbox("Descending", value=False if sort_col != "achieved_probability" else True)
        cand_pd = cand_pd.sort_values(sort_col, ascending=not descending)
        st.dataframe(cand_pd, use_container_width=True, hide_index=True)
        _download_csv_button(cand_pd, "Download candidate family CSV", "candidate_family.csv", "dl_candidates")

with tab5:
    st.subheader("Multi-Scenario View")
    st.caption("Select multiple scenarios and compare plans at one fixed probability.")
    if planning_mode == "advanced":
        st.info("Switch off advanced mode to compare multiple scenarios.")
    default_multi = SCENARIO_KEYS[: min(4, len(SCENARIO_KEYS))]
    selected_scenarios = st.multiselect("Scenarios", SCENARIO_KEYS, default=default_multi, format_func=lambda x: SCENARIO_LABELS.get(x, x))
    fixed_probability_text = st.text_input("Fixed probability for comparison", value=desired_probability_text, key="multi_prob")

    try:
        fixed_probability = _normalize_probability_text(fixed_probability_text.replace("%", ""))
        fixed_prob_error = None
    except Exception as exc:
        fixed_probability = None
        fixed_prob_error = str(exc)

    if fixed_prob_error:
        st.error(fixed_prob_error)
    elif planning_mode == "advanced":
        st.info("Multi-scenario comparison currently uses the simple TG-range mode only.")
    elif not simple_valid:
        st.info("Set a valid TG range to compare scenarios.")
    elif not selected_scenarios:
        st.info("Select at least one scenario.")
    else:
        rows = []
        for scenario_key in selected_scenarios:
            try:
                scenario_totals = _lookup_totals_simple(start_tg_level, target_tg_level, scenario_key)
                scenario_result = _optimal_plan_simple_cached(start_tg_level=start_tg_level, target_tg_level=target_tg_level, scenario=scenario_key, weeks=weeks, desired_probability=fixed_probability, current_ttg=current_ttg, used_this_week=used_this_week, last_week_days=last_week_days)
                row = {
                    "Scenario": SCENARIO_LABELS.get(scenario_key, scenario_key),
                    "Scenario ID": scenario_key,
                    "Start TG": start_tg_level,
                    "Target TG Level": target_tg_level,
                    "Building Cost (TG)": scenario_totals["target_tg"],
                    "Target TTG": scenario_totals["target_ttg"],
                    "Requested Probability": fixed_probability,
                }
                if scenario_result is None:
                    row["Status"] = "Unachievable"
                    for i in range(1, weeks + 1):
                        row[f"Week {i}"] = None
                    row["Total Refines"] = None
                    row["Refinement Cost (TG)"] = None
                    row["Total TG"] = None
                    row["Achieved Probability"] = None
                else:
                    row["Status"] = "OK"
                    for i, refs in enumerate(scenario_result.week_refines, start=1):
                        row[f"Week {i}"] = refs
                    row["Total Refines"] = scenario_result.total_refines
                    row["Refinement Cost (TG)"] = scenario_result.total_cost
                    row["Total TG"] = scenario_totals["target_tg"] + scenario_result.total_cost
                    row["Achieved Probability"] = scenario_result.achieved_probability
                rows.append(row)
            except Exception:
                row = {
                    "Scenario": SCENARIO_LABELS.get(scenario_key, scenario_key),
                    "Scenario ID": scenario_key,
                    "Start TG": start_tg_level,
                    "Target TG Level": target_tg_level,
                    "Building Cost (TG)": None,
                    "Target TTG": None,
                    "Requested Probability": fixed_probability,
                    "Status": "Error",
                }
                for i in range(1, weeks + 1):
                    row[f"Week {i}"] = None
                row["Total Refines"] = None
                row["Refinement Cost (TG)"] = None
                row["Total TG"] = None
                row["Achieved Probability"] = None
                rows.append(row)
        if rows:
            multi_df = pd.DataFrame(rows)
            st.dataframe(multi_df, use_container_width=True, hide_index=True)
            _download_csv_button(multi_df, "Download multi-scenario CSV", "multi_scenario_view.csv", "dl_multi")

with tab6:
    st.subheader("Validation")
    if planning_mode is None:
        st.info("Enter a valid target configuration to view validation details.")
    elif planning_mode == "advanced":
        val_df = _validation_panel_advanced(building_inputs or {}, target_tg, target_ttg)

        forced_df = val_df[val_df["forced_by_prereq"] == True].copy()
        if not forced_df.empty:
            st.markdown("### Forced prerequisite updates")
            st.dataframe(
                forced_df[["building", "start", "requested_target", "final_target", "tg_cost", "ttg_cost"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Full validation table")
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        _download_csv_button(val_df, "Download validation CSV", "validation_advanced.csv", "dl_validation_adv")
    else:
        val_df = _simple_scenario_summary(start_tg_level, target_tg_level, scenario)
        val_df["building_cost_tg_total"] = target_tg
        val_df["target_ttg_total"] = target_ttg

        forced_df = val_df[val_df["forced_by_prereq"] == True].copy()
        if not forced_df.empty:
            st.markdown("### Forced prerequisite updates")
            st.dataframe(
                forced_df[["building", "start", "requested_target", "final_target", "tg_cost", "ttg_cost"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Scenario validation")
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        _download_csv_button(val_df, "Download validation CSV", "validation_simple.csv", "dl_validation_simple")
