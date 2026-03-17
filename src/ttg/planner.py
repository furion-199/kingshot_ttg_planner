from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import ceil
from typing import Iterable
import time

import numpy as np

from ttg.cost_lookup import lookup_target_ttg
from ttg.cost_model import cost_additional_this_week, cost_one_week, normalize_last_week_days
from ttg.probabilities import get_tier, tier_distribution, tier_expected_value, tier_max, tier_min


EPS = 1e-10
PMF_ROUND_DECIMALS = 12

# Frontier controls
DEFAULT_PROB_BUCKET = 0.005
DEFAULT_MAX_FRONTIER_STATES = 800
EXACT_FRONTIER_WEEK_LIMIT = 3
KEEP_STATES_PER_BUCKET = 2


@dataclass(frozen=True)
class PlanResult:
    target_ttg: int
    current_ttg: int
    used_this_week: int
    weeks: int
    requested_probability: float | None
    achieved_probability: float
    total_refines: int
    total_cost: float
    week_refines: tuple[int, ...]


@dataclass
class _State:
    pmf: np.ndarray
    survival: np.ndarray
    cost: float
    total_refines: int
    plan: tuple[int, ...]
    next_max: int
    achieved_probability: float


@dataclass(frozen=True)
class _WeekOption:
    refines: int
    cost: float
    pmf: np.ndarray
    survival: np.ndarray


def normalize_probability_input(desired_prob: float | int | None) -> float | None:
    if desired_prob is None:
        return None

    p = float(desired_prob)
    if p > 1.0:
        p = p / 100.0

    if p <= 0.0 or p > 1.0:
        raise ValueError("desired_prob must be in (0,1] or (0,100]")

    return p


def week_refine_capacity(week_index: int, used_this_week: int) -> int:
    return 100 - used_this_week if week_index == 1 else 100


def ttg_expected_one_week(refines: int, *, used_this_week: int = 0) -> float:
    total = 0.0
    for i in range(1, refines + 1):
        tier = get_tier(used_this_week + i)
        total += tier_expected_value(tier)
    return total


def ttg_min_one_week(refines: int, *, used_this_week: int = 0) -> int:
    total = 0
    for i in range(1, refines + 1):
        tier = get_tier(used_this_week + i)
        total += tier_min(tier)
    return total


def ttg_max_one_week(refines: int, *, used_this_week: int = 0) -> int:
    total = 0
    for i in range(1, refines + 1):
        tier = get_tier(used_this_week + i)
        total += tier_max(tier)
    return total


@lru_cache(maxsize=None)
def _week_full_distribution(refines: int, used_this_week: int = 0) -> np.ndarray:
    if refines < 0:
        raise ValueError("refines must be >= 0")
    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")
    if used_this_week + refines > 100:
        raise ValueError("used_this_week + refines cannot exceed 100")

    dist = np.array([1.0], dtype=np.float64)

    for i in range(1, refines + 1):
        tier = get_tier(used_this_week + i)
        yields, probs = tier_distribution(tier)

        next_len = len(dist) + int(yields.max())
        next_dist = np.zeros(next_len, dtype=np.float64)

        for y, p in zip(yields, probs, strict=False):
            next_dist[y : y + len(dist)] += dist * p

        dist = next_dist

    return dist


@lru_cache(maxsize=None)
def _week_tail_distribution(refines: int, target_ttg: int, used_this_week: int = 0) -> np.ndarray:
    full = _week_full_distribution(refines, used_this_week)

    result = np.zeros(target_ttg + 1, dtype=np.float64)

    if len(full) <= target_ttg + 1:
        result[: len(full)] = full
    else:
        result[:target_ttg] = full[:target_ttg]
        result[target_ttg] = full[target_ttg:].sum()

    return result


def _survival_from_pmf(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(pmf[::-1])[::-1]


def _convolve_tail(a: np.ndarray, b: np.ndarray, target_ttg: int) -> np.ndarray:
    full = np.convolve(a, b)
    result = np.zeros(target_ttg + 1, dtype=np.float64)

    if len(full) <= target_ttg + 1:
        result[: len(full)] = full
    else:
        result[:target_ttg] = full[:target_ttg]
        result[target_ttg] = full[target_ttg:].sum()

    return result


def _dist_pow_tail(base: np.ndarray, exponent: int, target_ttg: int) -> np.ndarray:
    result = np.zeros(target_ttg + 1, dtype=np.float64)
    result[0] = 1.0

    if exponent == 0:
        return result

    power = base.copy()

    while exponent > 0:
        if exponent & 1:
            result = _convolve_tail(result, power, target_ttg)

        exponent //= 2

        if exponent > 0:
            power = _convolve_tail(power, power, target_ttg)

    return result


@lru_cache(maxsize=None)
def _week_option_catalog(target_ttg: int, used_this_week: int) -> tuple[_WeekOption, ...]:
    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")

    cap = 100 - used_this_week
    options: list[_WeekOption] = []

    for refs in range(cap + 1):
        pmf = _week_tail_distribution(refs, target_ttg, used_this_week)
        survival = _survival_from_pmf(pmf)

        if used_this_week == 0:
            cost = cost_one_week(refs)
        else:
            cost = cost_additional_this_week(refs, used_this_week)

        options.append(
            _WeekOption(
                refines=refs,
                cost=float(cost),
                pmf=pmf,
                survival=survival,
            )
        )

    return tuple(options)


def probability_for_even_distribution(
    target_ttg: int,
    refines: int,
    weeks: int | None = None,
) -> float:
    if target_ttg <= 0:
        return 1.0
    if refines <= 0:
        return 0.0

    if weeks is None:
        weeks = ceil(refines / 100)

    if weeks <= 0:
        raise ValueError("weeks must be > 0")
    if refines > 100 * weeks:
        raise ValueError("refines exceed weekly capacity")

    base = refines // weeks
    extra = refines % weeks

    total_min = (weeks - extra) * ttg_min_one_week(base)
    total_max = (weeks - extra) * ttg_max_one_week(base)

    if extra > 0:
        total_min += extra * ttg_min_one_week(base + 1)
        total_max += extra * ttg_max_one_week(base + 1)

    if target_ttg <= total_min:
        return 1.0
    if target_ttg > total_max:
        return 0.0

    tail_small = _week_tail_distribution(base, target_ttg, 0)
    combined = _dist_pow_tail(tail_small, weeks - extra, target_ttg)

    if extra > 0:
        tail_large = _week_tail_distribution(base + 1, target_ttg, 0)
        combined = _convolve_tail(
            combined,
            _dist_pow_tail(tail_large, extra, target_ttg),
            target_ttg,
        )

    return float(combined[target_ttg])


def probability_for_plan(
    target_ttg: int,
    plan: Iterable[int],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
) -> float:
    remaining_target = target_ttg - current_ttg
    if remaining_target <= 0:
        return 1.0

    dist = np.zeros(remaining_target + 1, dtype=np.float64)
    dist[0] = 1.0

    for week_index, refs in enumerate(plan, start=1):
        if refs < 0:
            raise ValueError("weekly refines must be >= 0")

        week_used = used_this_week if week_index == 1 else 0
        cap = week_refine_capacity(week_index, used_this_week)
        if refs > cap:
            raise ValueError(f"week {week_index} exceeds capacity ({cap})")

        week_tail = _week_tail_distribution(refs, remaining_target, week_used)
        dist = _convolve_tail(dist, week_tail, remaining_target)

    return float(dist[remaining_target])


def cost_for_plan(
    plan: Iterable[int],
    *,
    used_this_week: int = 0,
    last_week_days: int = 5,
) -> float:
    plan_tuple = tuple(int(x) for x in plan)

    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")

    last_week_days = normalize_last_week_days(last_week_days)
    last_week_discount_refines = last_week_days

    total = 0.0
    n = len(plan_tuple)

    for week_index, refs in enumerate(plan_tuple, start=1):
        if refs < 0:
            raise ValueError("weekly refines must be >= 0")

        is_first = week_index == 1
        is_last = week_index == n

        if is_first:
            if refs > 100 - used_this_week:
                raise ValueError("week 1 exceeds remaining weekly capacity")

            discount_refines = last_week_discount_refines if is_last else 7
            total += cost_additional_this_week(
                refs,
                used_this_week,
                discount_refines=discount_refines,
            )
        else:
            if refs > 100:
                raise ValueError("week exceeds weekly capacity")

            discount_refines = last_week_discount_refines if is_last else 7
            total += cost_one_week(
                refs,
                discount_refines=discount_refines,
            )

    return total


def _plan_tiebreak_key(plan: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-x for x in plan)


def _state_sort_key(state: _State) -> tuple:
    return (
        state.cost,
        state.total_refines,
        _plan_tiebreak_key(state.plan),
        -state.achieved_probability,
        -state.next_max,
    )


def _state_dominates(a: _State, b: _State) -> bool:
    if a.cost > b.cost + EPS:
        return False
    if a.total_refines > b.total_refines:
        return False
    if a.next_max < b.next_max:
        return False
    if a.achieved_probability + EPS < b.achieved_probability:
        return False
    if np.any(a.survival + EPS < b.survival):
        return False
    return True


def all_in_plan(weeks: int, used_this_week: int = 0) -> tuple[int, ...]:
    """
    Max-refine front-loaded plan for the given horizon.
    """
    if weeks <= 0:
        raise ValueError("weeks must be > 0")
    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")

    first_cap = 100 - used_this_week
    return (first_cap,) + tuple(100 for _ in range(weeks - 1))


def plan_to_dict(result: PlanResult) -> dict:
    out = asdict(result)
    out["week_refines"] = list(result.week_refines)
    return out


def plan_table_rows(
    tg_level: int,
    scenario: str,
    weeks: int,
    probabilities: Iterable[float | int],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    prob_bucket: float = DEFAULT_PROB_BUCKET,  # kept for API compatibility; unused
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,  # kept for API compatibility; unused
) -> list[dict]:
    target_ttg = lookup_target_ttg(tg_level, scenario)

    requested_probs = [normalize_probability_input(p) for p in probabilities]
    ordered = sorted(enumerate(requested_probs), key=lambda x: x[1], reverse=True)

    solved: dict[int, PlanResult] = {}
    prior_plans: list[tuple[int, ...]] = []

    for idx, req in ordered:
        result = _solve_plan_heuristic(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=req,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            extra_seed_plans=prior_plans,
        )
        solved[idx] = result
        prior_plans.append(result.week_refines)

    rows: list[dict] = []

    for idx, req in enumerate(requested_probs):
        result = solved[idx]

        row = {
            "tg_level": tg_level,
            "scenario": scenario,
            "requested_probability": req,
            "target_ttg": result.target_ttg,
            "current_ttg": result.current_ttg,
            "used_this_week": result.used_this_week,
            "total_refines": result.total_refines,
            "total_cost": result.total_cost,
            "achieved_probability": result.achieved_probability,
        }

        for i, refs in enumerate(result.week_refines, start=1):
            row[f"week_{i}"] = refs

        rows.append(row)

    return rows

def evaluate_plan(
    target_ttg: int,
    week_refines: Iterable[int],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
) -> dict:
    """
    Return probability and TG cost for a custom plan.
    """
    plan_tuple = tuple(int(x) for x in week_refines)
    last_week_days = normalize_last_week_days(last_week_days)

    return {
        "target_ttg": int(target_ttg),
        "current_ttg": int(current_ttg),
        "used_this_week": int(used_this_week),
        "last_week_days": int(last_week_days),
        "weeks": len(plan_tuple),
        "week_refines": list(plan_tuple),
        "total_refines": int(sum(plan_tuple)),
        "total_cost": float(
            cost_for_plan(
                plan=plan_tuple,
                used_this_week=used_this_week,
                last_week_days=last_week_days,
            )
        ),
        "achieved_probability": float(
            probability_for_plan(
                target_ttg=target_ttg,
                plan=plan_tuple,
                current_ttg=current_ttg,
                used_this_week=used_this_week,
            )
        ),
    }


def evaluate_plan_from_lookup(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    week_refines: Iterable[int],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
) -> dict:
    """
    Return target TTG, probability, and TG cost for a custom plan using lookup target.
    """
    target_ttg = lookup_target_ttg(start_tg_level, target_tg_level, scenario)

    result = evaluate_plan(
        target_ttg=target_ttg,
        week_refines=week_refines,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )

    result["start_tg_level"] = int(start_tg_level)
    result["target_tg_level"] = int(target_tg_level)
    result["scenario"] = scenario

    return result


def evaluate_plans_table(
    target_ttg: int,
    plans: list[Iterable[int]],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
) -> list[dict]:
    """
    Evaluate multiple custom plans at once.
    """
    return [
        evaluate_plan(
            target_ttg=target_ttg,
            week_refines=plan,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
        )
        for plan in plans
    ]

def all_in_plan(weeks: int, used_this_week: int) -> tuple[int, ...]:
    first_cap = 100 - used_this_week
    return (first_cap,) + tuple(100 for _ in range(weeks - 1))


def _canonicalize_plan(plan: Iterable[int], used_this_week: int) -> tuple[int, ...]:
    vals = [int(x) for x in plan]
    if not vals:
        return tuple()

    if used_this_week > 0:
        return (vals[0],) + tuple(sorted(vals[1:], reverse=True))

    return tuple(sorted(vals, reverse=True))


def even_split_plan(
    total_refines: int,
    weeks: int,
    used_this_week: int = 0,
    *,
    front_load_remainder: bool = True,
) -> tuple[int, ...]:
    """
    Near-even split that respects week caps.
    Implemented by round-robin fill, which is robust and simple.
    """
    if total_refines < 0:
        raise ValueError("total_refines must be >= 0")

    capacities = [100 - used_this_week] + [100] * (weeks - 1)
    if total_refines > sum(capacities):
        raise ValueError("total_refines exceeds total capacity")

    plan = [0] * weeks
    remaining = total_refines

    order = list(range(weeks))
    if not front_load_remainder:
        order = list(reversed(order))

    while remaining > 0:
        progressed = False
        for idx in order:
            if remaining == 0:
                break
            if plan[idx] < capacities[idx]:
                plan[idx] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return _canonicalize_plan(plan, used_this_week)


def _result_sort_key(result: PlanResult) -> tuple:
    return (
        result.total_cost,
        result.total_refines,
        tuple(-x for x in result.week_refines),
        -result.achieved_probability,
    )


def _make_plan_result(
    target_ttg: int,
    plan: tuple[int, ...],
    *,
    current_ttg: int,
    used_this_week: int,
    requested_probability: float | None,
    last_week_days: int,
) -> PlanResult:
    eval_row = evaluate_plan(
        target_ttg=target_ttg,
        week_refines=plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )

    return PlanResult(
        target_ttg=target_ttg,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        weeks=len(plan),
        requested_probability=requested_probability,
        achieved_probability=float(eval_row["achieved_probability"]),
        total_refines=int(eval_row["total_refines"]),
        total_cost=float(eval_row["total_cost"]),
        week_refines=tuple(int(x) for x in plan),
    )


def _max_total_refines(weeks: int, used_this_week: int) -> int:
    return (100 - used_this_week) + 100 * (weeks - 1)


def _compress_plan_front_loaded(plan: tuple[int, ...], move: int, used_this_week: int) -> tuple[int, ...]:
    vals = list(plan)
    caps = [100 - used_this_week] + [100] * (len(vals) - 1)

    for j in range(len(vals) - 1, 0, -1):
        if vals[j] <= 0:
            continue

        delta = min(move, vals[j], caps[0] - vals[0])
        if delta <= 0:
            continue

        vals[j] -= delta
        vals[0] += delta
        return _canonicalize_plan(vals, used_this_week)

    return plan


def _spread_plan(plan: tuple[int, ...], move: int, used_this_week: int) -> tuple[int, ...]:
    vals = list(plan)
    caps = [100 - used_this_week] + [100] * (len(vals) - 1)

    for i in range(len(vals)):
        if vals[i] < move:
            continue

        for j in range(len(vals) - 1, -1, -1):
            if i == j:
                continue
            if vals[j] + move > caps[j]:
                continue

            vals2 = vals.copy()
            vals2[i] -= move
            vals2[j] += move
            return _canonicalize_plan(vals2, used_this_week)

    return plan


def _two_level_plan(
    total_refines: int,
    weeks: int,
    high_weeks: int,
    *,
    used_this_week: int = 0,
) -> tuple[int, ...]:
    """
    Split total refines across two levels:
      first `high_weeks` weeks get the higher level,
      remaining weeks get the lower level,
    then canonicalize.

    This creates plans like:
      [40,40,40,38,38,38,38,38]
    """
    if weeks <= 0:
        raise ValueError("weeks must be > 0")
    if high_weeks < 0 or high_weeks > weeks:
        raise ValueError("high_weeks must be between 0 and weeks")

    caps = [100 - used_this_week] + [100] * (weeks - 1)
    if total_refines < 0 or total_refines > sum(caps):
        raise ValueError("total_refines out of range")

    base = total_refines // weeks
    extra = total_refines % weeks

    vals = [base] * weeks

    # Put extra into the early "high" weeks first
    for i in range(min(extra, high_weeks)):
        vals[i] += 1

    remaining_extra = extra - min(extra, high_weeks)

    # If extra still remains, distribute to later weeks
    for i in range(high_weeks, high_weeks + remaining_extra):
        if i < weeks:
            vals[i] += 1

    # Respect caps conservatively
    vals = [min(vals[i], caps[i]) for i in range(weeks)]

    # If capping lost some refines, redistribute greedily
    deficit = total_refines - sum(vals)
    if deficit > 0:
        for i in range(weeks):
            add = min(deficit, caps[i] - vals[i])
            vals[i] += add
            deficit -= add
            if deficit == 0:
                break

    return _canonicalize_plan(vals, used_this_week)


def _stair_plan(
    total_refines: int,
    weeks: int,
    *,
    used_this_week: int = 0,
    drop_every: int = 2,
) -> tuple[int, ...]:
    """
    Create a mild stair-step / tapered plan around the even split.

    Example shape before canonicalization:
      [40,40,39,39,38,38,37,37]
    """
    if weeks <= 0:
        raise ValueError("weeks must be > 0")

    caps = [100 - used_this_week] + [100] * (weeks - 1)
    if total_refines < 0 or total_refines > sum(caps):
        raise ValueError("total_refines out of range")

    base = total_refines // weeks
    vals = [base] * weeks

    # Build a mild taper
    offset = 0
    for i in range(weeks):
        vals[i] += max(0, 2 - (i // max(1, drop_every)))
        offset += max(0, 2 - (i // max(1, drop_every)))

    # Bring back to target total
    over = sum(vals) - total_refines
    i = weeks - 1
    while over > 0 and i >= 0:
        take = min(over, vals[i])
        vals[i] -= take
        over -= take
        i -= 1

    # If we undershot, refill respecting caps
    under = total_refines - sum(vals)
    i = 0
    while under > 0 and i < weeks:
        add = min(under, caps[i] - vals[i])
        vals[i] += add
        under -= add
        i += 1

    vals = [min(vals[i], caps[i]) for i in range(weeks)]

    # Final top-up if caps caused undershoot
    under = total_refines - sum(vals)
    if under > 0:
        for i in range(weeks):
            add = min(under, caps[i] - vals[i])
            vals[i] += add
            under -= add
            if under == 0:
                break

    return _canonicalize_plan(vals, used_this_week)

def _build_plan_from_week1_and_later_multiset(
    week1: int,
    later_multiset: dict[int, int],
    *,
    weeks: int,
    used_this_week: int,
) -> tuple[int, ...]:
    """
    Build a canonical plan from:
      - explicit week1
      - multiset for weeks 2..N

    later_multiset example:
      {38: 4, 37: 3}
    """
    first_cap = 100 - used_this_week
    if week1 < 0 or week1 > first_cap:
        raise ValueError("week1 out of range")

    later: list[int] = []
    for value, count in later_multiset.items():
        if value < 0 or value > 100:
            raise ValueError("later-week refine count out of range")
        if count < 0:
            raise ValueError("multiset counts must be >= 0")
        later.extend([int(value)] * int(count))

    if len(later) != weeks - 1:
        raise ValueError("later multiset size does not match remaining weeks")

    later.sort(reverse=True)
    plan = (int(week1), *later)
    return _canonicalize_plan(plan, used_this_week)


def _later_equal_multisets(
    later_total: int,
    later_weeks: int,
) -> list[dict[int, int]]:
    """
    Later weeks all equal or as equal as possible.
    """
    if later_weeks <= 0:
        return [{}]

    base = later_total // later_weeks
    extra = later_total % later_weeks

    out: list[dict[int, int]] = []

    d: dict[int, int] = {}
    if extra > 0:
        d[base + 1] = extra
    if later_weeks - extra > 0:
        d[base] = later_weeks - extra
    out.append(d)

    return out


def _later_two_level_multisets(
    later_total: int,
    later_weeks: int,
    *,
    max_gap: int = 2,
) -> list[dict[int, int]]:
    """
    Later weeks use two nearby values:
      hi repeated a times
      lo repeated b times
    with hi - lo <= max_gap
    """
    if later_weeks <= 0:
        return [{}]

    out: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    avg = later_total / later_weeks
    lo_center = int(avg)

    for gap in range(1, max_gap + 1):
        for lo in range(max(0, lo_center - 2), lo_center + 3):
            hi = lo + gap

            # Solve: a*hi + (later_weeks-a)*lo = later_total
            denom = hi - lo
            num = later_total - later_weeks * lo

            if denom == 0:
                continue
            if num < 0 or num > later_weeks * denom:
                continue
            if num % denom != 0:
                continue

            a = num // denom
            b = later_weeks - a

            d: dict[int, int] = {}
            if a > 0:
                d[hi] = a
            if b > 0:
                d[lo] = b

            key = tuple(sorted(d.items(), reverse=True))
            if key not in seen:
                seen.add(key)
                out.append(d)

    return out


def _later_three_level_multisets(
    later_total: int,
    later_weeks: int,
) -> list[dict[int, int]]:
    """
    Later weeks use three adjacent values around the mean:
      hi, mid, lo
    where hi = mid+1 and lo = mid-1
    """
    if later_weeks <= 0:
        return [{}]

    out: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    avg = later_total / later_weeks
    mid_center = int(round(avg))

    for mid in range(max(0, mid_center - 2), mid_center + 3):
        lo = max(0, mid - 1)
        hi = mid + 1

        # a*hi + b*mid + c*lo = later_total
        # a+b+c = later_weeks
        for a in range(later_weeks + 1):
            for b in range(later_weeks - a + 1):
                c = later_weeks - a - b
                total = a * hi + b * mid + c * lo
                if total != later_total:
                    continue

                d: dict[int, int] = {}
                if a > 0:
                    d[hi] = a
                if b > 0:
                    d[mid] = b
                if c > 0:
                    d[lo] = c

                key = tuple(sorted(d.items(), reverse=True))
                if key not in seen:
                    seen.add(key)
                    out.append(d)

    return out


def _week1_later_multiset_candidates(
    total_refines: int,
    weeks: int,
    *,
    used_this_week: int,
) -> list[tuple[int, ...]]:
    """
    Generate candidates by explicitly choosing week1, then filling weeks 2..N
    from small-support multisets.
    """
    first_cap = 100 - used_this_week
    later_weeks = weeks - 1

    if weeks <= 1:
        return [(min(total_refines, first_cap),)]

    out: set[tuple[int, ...]] = set()

    # Try week1 near the average and also slightly front-loaded.
    avg = total_refines / weeks
    week1_guesses = {
        int(avg),
        int(avg) + 1,
        int(avg) + 2,
        int(avg) + 5,
        min(first_cap, int(avg) + 10),
        min(first_cap, total_refines),
    }

    for week1 in sorted(week1_guesses):
        if week1 < 0 or week1 > first_cap:
            continue

        later_total = total_refines - week1
        if later_total < 0 or later_total > 100 * later_weeks:
            continue

        families: list[dict[int, int]] = []
        families.extend(_later_equal_multisets(later_total, later_weeks))
        families.extend(_later_two_level_multisets(later_total, later_weeks, max_gap=2))
        families.extend(_later_three_level_multisets(later_total, later_weeks))

        for fam in families:
            try:
                out.add(
                    _build_plan_from_week1_and_later_multiset(
                        week1,
                        fam,
                        weeks=weeks,
                        used_this_week=used_this_week,
                    )
                )
            except Exception:
                pass

    return sorted(out, key=lambda p: (sum(p), p))

    
def _candidate_plans_for_total(
    total_refines: int,
    weeks: int,
    *,
    used_this_week: int,
) -> list[tuple[int, ...]]:
    """
    Structured candidate family for a given total.

    Includes:
    - even split (front/back remainder)
    - compressed/spread variants
    - two-level variants
    - stair-step variants
    - multiset-based later-week variants

    This is intentionally a UNION of families, not a replacement.
    """
    plans: set[tuple[int, ...]] = set()

    # ---- baseline even families ----
    even_front = even_split_plan(
        total_refines,
        weeks,
        used_this_week,
        front_load_remainder=True,
    )
    even_back = even_split_plan(
        total_refines,
        weeks,
        used_this_week,
        front_load_remainder=False,
    )

    plans.add(even_front)
    plans.add(even_back)

    for move in (1, 2, 5):
        plans.add(_compress_plan_front_loaded(even_front, move, used_this_week))
        plans.add(_compress_plan_front_loaded(even_back, move, used_this_week))
        plans.add(_spread_plan(even_front, move, used_this_week))
        plans.add(_spread_plan(even_back, move, used_this_week))

    # ---- previous two-level families ----
    for high_weeks in {
        1,
        2,
        max(1, weeks // 4),
        max(1, weeks // 2),
        max(1, (3 * weeks) // 4),
        weeks - 1,
        weeks,
    }:
        if 0 <= high_weeks <= weeks:
            try:
                plans.add(
                    _two_level_plan(
                        total_refines,
                        weeks,
                        high_weeks,
                        used_this_week=used_this_week,
                    )
                )
            except Exception:
                pass

    # ---- previous stair-step families ----
    for drop_every in (1, 2, 3):
        try:
            plans.add(
                _stair_plan(
                    total_refines,
                    weeks,
                    used_this_week=used_this_week,
                    drop_every=drop_every,
                )
            )
        except Exception:
            pass

    # ---- new multiset later-week families ----
    try:
        for plan in _week1_later_multiset_candidates(
            total_refines,
            weeks,
            used_this_week=used_this_week,
        ):
            plans.add(plan)
    except Exception:
        pass

    return sorted(plans, key=lambda p: (sum(p), p))


def _best_feasible_candidate_for_total(
    target_ttg: int,
    total_refines: int,
    weeks: int,
    goal_prob: float,
    *,
    current_ttg: int,
    used_this_week: int,
    requested_probability: float | None,
    last_week_days: int,
    eval_cache: dict[tuple[int, ...], PlanResult],
) -> PlanResult | None:
    best: PlanResult | None = None

    for plan in _candidate_plans_for_total(
        total_refines,
        weeks,
        used_this_week=used_this_week,
    ):
        if plan not in eval_cache:
            eval_cache[plan] = _make_plan_result(
                target_ttg=target_ttg,
                plan=plan,
                current_ttg=current_ttg,
                used_this_week=used_this_week,
                requested_probability=requested_probability,
                last_week_days=last_week_days,
            )

        result = eval_cache[plan]

        if result.achieved_probability + EPS < goal_prob:
            continue

        if best is None or _result_sort_key(result) < _result_sort_key(best):
            best = result

    return best


def _first_feasible_total_via_candidates(
    target_ttg: int,
    weeks: int,
    goal_prob: float,
    *,
    current_ttg: int,
    used_this_week: int,
    requested_probability: float | None,
    eval_cache: dict[tuple[int, ...], PlanResult],
) -> tuple[int, PlanResult]:
    lo = 0
    hi = _max_total_refines(weeks, used_this_week)

    best_total = hi
    best_result: PlanResult | None = None

    while lo <= hi:
        mid = (lo + hi) // 2

        feasible = _best_feasible_candidate_for_total(
            target_ttg=target_ttg,
            total_refines=mid,
            weeks=weeks,
            goal_prob=goal_prob,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            requested_probability=requested_probability,
            eval_cache=eval_cache,
        )

        if feasible is not None:
            best_total = mid
            best_result = feasible
            hi = mid - 1
        else:
            lo = mid + 1

    if best_result is None:
        raise ValueError("Desired probability is unattainable with available weeks/capacity")

    return best_total, best_result


def _neighbor_plans_small(
    plan: tuple[int, ...],
    *,
    used_this_week: int,
    steps: tuple[int, ...] = (5, 2, 1),
) -> list[tuple[int, ...]]:
    weeks = len(plan)
    caps = [100 - used_this_week] + [100] * (weeks - 1)

    out: set[tuple[int, ...]] = set()

    for step in steps:
        # reduce one week
        for i in range(weeks):
            if plan[i] >= step:
                vals = list(plan)
                vals[i] -= step
                out.add(_canonicalize_plan(vals, used_this_week))

        # transfer
        for i in range(weeks):
            if plan[i] < step:
                continue
            for j in range(weeks):
                if i == j:
                    continue
                if plan[j] + step > caps[j]:
                    continue

                vals = list(plan)
                vals[i] -= step
                vals[j] += step
                out.add(_canonicalize_plan(vals, used_this_week))

    out.discard(plan)
    return list(out)


def _local_cleanup(
    start_result: PlanResult,
    target_ttg: int,
    goal_prob: float,
    *,
    current_ttg: int,
    used_this_week: int,
    requested_probability: float | None,
    last_week_days: int,
    eval_cache: dict[tuple[int, ...], PlanResult],
) -> PlanResult:
    current = start_result

    for step_set in ((5, 2), (2, 1), (1,)):
        improved = True

        while improved:
            improved = False
            best = current

            for cand in _neighbor_plans_small(
                current.week_refines,
                used_this_week=used_this_week,
                steps=step_set,
            ):
                if cand not in eval_cache:
                    eval_cache[cand] = _make_plan_result(
                        target_ttg=target_ttg,
                        plan=cand,
                        current_ttg=current_ttg,
                        used_this_week=used_this_week,
                        requested_probability=requested_probability,
                        last_week_days=last_week_days,
                    )

                res = eval_cache[cand]

                if res.achieved_probability + EPS < goal_prob:
                    continue

                if _result_sort_key(res) < _result_sort_key(best):
                    best = res

            if _result_sort_key(best) < _result_sort_key(current):
                current = best
                improved = True

    return current

def _first_feasible_even_total(
    target_ttg: int,
    weeks: int,
    goal_prob: float,
    *,
    current_ttg: int,
    used_this_week: int,
    front_load_remainder: bool,
) -> int:
    lo = 0
    hi = _max_total_refines(weeks, used_this_week)

    while lo < hi:
        mid = (lo + hi) // 2
        plan = even_split_plan(
            mid,
            weeks,
            used_this_week,
            front_load_remainder=front_load_remainder,
        )
        prob = probability_for_plan(
            target_ttg=target_ttg,
            plan=plan,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
        )

        if prob + EPS >= goal_prob:
            hi = mid
        else:
            lo = mid + 1

    return lo


def _solve_plan_heuristic(
    target_ttg: int,
    weeks: int,
    desired_probability: float | None,
    *,
    current_ttg: int,
    used_this_week: int,
    last_week_days: int,
    extra_seed_plans: Iterable[Iterable[int]] = (),
) -> PlanResult:
    if target_ttg <= 0:
        raise ValueError("target_ttg must be > 0")
    if weeks <= 0:
        raise ValueError("weeks must be > 0")
    if current_ttg < 0:
        raise ValueError("current_ttg must be >= 0")
    if used_this_week < 0 or used_this_week > 100:
        raise ValueError("used_this_week must be between 0 and 100")

    last_week_days = normalize_last_week_days(last_week_days)

    eval_cache: dict[tuple[int, ...], PlanResult] = {}

    all_in = all_in_plan(weeks, used_this_week)
    all_in_result = _make_plan_result(
        target_ttg=target_ttg,
        plan=all_in,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        requested_probability=desired_probability,
        last_week_days=last_week_days,
    )
    eval_cache[all_in] = all_in_result

    if desired_probability is None:
        if all_in_result.achieved_probability < 1.0 - EPS:
            return all_in_result
        goal = 1.0
    else:
        goal = desired_probability

    if all_in_result.achieved_probability + EPS < goal:
        raise ValueError("Desired probability is unattainable with available weeks/capacity")

    base_total = _first_feasible_even_total(
        target_ttg=target_ttg,
        weeks=weeks,
        goal_prob=goal,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        front_load_remainder=True,
    )

    max_total = _max_total_refines(weeks, used_this_week)
    best = all_in_result

    for seed in extra_seed_plans:
        plan = _canonicalize_plan(seed, used_this_week)
        if plan not in eval_cache:
            eval_cache[plan] = _make_plan_result(
                target_ttg=target_ttg,
                plan=plan,
                current_ttg=current_ttg,
                used_this_week=used_this_week,
                requested_probability=desired_probability,
                last_week_days=last_week_days,
            )

        result = eval_cache[plan]
        if result.achieved_probability + EPS >= goal:
            if _result_sort_key(result) < _result_sort_key(best):
                best = result

    coarse_offsets = (0, 1, 2, 5, 10, 20, 40, 80)
    totals_to_try = sorted(
        {
            total
            for total in (base_total + off for off in coarse_offsets)
            if base_total <= total <= max_total
        }
    )

    for total in totals_to_try:
        candidate = _best_feasible_candidate_for_total(
            target_ttg=target_ttg,
            total_refines=total,
            weeks=weeks,
            goal_prob=goal,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            requested_probability=desired_probability,
            last_week_days=last_week_days,
            eval_cache=eval_cache,
        )

        if candidate is not None and _result_sort_key(candidate) < _result_sort_key(best):
            best = candidate

    best_total = best.total_refines
    local_min = max(base_total, best_total - 10)
    local_max = min(max_total, best_total + 10)

    for total in range(local_min, local_max + 1):
        candidate = _best_feasible_candidate_for_total(
            target_ttg=target_ttg,
            total_refines=total,
            weeks=weeks,
            goal_prob=goal,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            requested_probability=desired_probability,
            last_week_days=last_week_days,
            eval_cache=eval_cache,
        )

        if candidate is not None and _result_sort_key(candidate) < _result_sort_key(best):
            best = candidate

    best = _local_cleanup(
        start_result=best,
        target_ttg=target_ttg,
        goal_prob=goal,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        requested_probability=desired_probability,
        last_week_days=last_week_days,
        eval_cache=eval_cache,
    )

    return best


def optimal_plan(
    target_ttg: int,
    weeks: int,
    desired_probability: float | int | None = None,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,  # kept for API compatibility; unused
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,  # kept for API compatibility; unused
) -> PlanResult:
    requested = normalize_probability_input(desired_probability)
    last_week_days = normalize_last_week_days(last_week_days)

    return _solve_plan_heuristic(
        target_ttg=target_ttg,
        weeks=weeks,
        desired_probability=requested,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )


def optimal_plan_from_lookup(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    desired_probability: float | int | None = None,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,  # kept for API compatibility; unused
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,  # kept for API compatibility; unused
) -> PlanResult:
    target_ttg = lookup_target_ttg(start_tg_level, target_tg_level, scenario)
    return optimal_plan(
        target_ttg=target_ttg,
        weeks=weeks,
        desired_probability=desired_probability,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
        prob_bucket=prob_bucket,
        max_frontier_states=max_frontier_states,
    )

def plan_table_rows(
    start_tg_level: int,
    target_tg_level: int,
    scenario: str,
    weeks: int,
    probabilities: Iterable[float | int],
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    prob_bucket: float = DEFAULT_PROB_BUCKET,  # kept for API compatibility; unused
    max_frontier_states: int = DEFAULT_MAX_FRONTIER_STATES,  # kept for API compatibility; unused
) -> list[dict]:
    target_ttg = lookup_target_ttg(start_tg_level, target_tg_level, scenario)
    last_week_days = normalize_last_week_days(last_week_days)

    requested_probs = [normalize_probability_input(p) for p in probabilities]
    ordered = sorted(enumerate(requested_probs), key=lambda x: x[1], reverse=True)

    solved: dict[int, PlanResult] = {}
    prior_plans: list[tuple[int, ...]] = []

    for idx, req in ordered:
        result = _solve_plan_heuristic(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=req,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
            extra_seed_plans=prior_plans,
        )
        solved[idx] = result
        prior_plans.append(result.week_refines)

    rows: list[dict] = []

    for idx, req in enumerate(requested_probs):
        result = solved[idx]

        row = {
            "start_tg_level": start_tg_level,
            "target_tg_level": target_tg_level,
            "scenario": scenario,
            "requested_probability": req,
            "target_ttg": result.target_ttg,
            "current_ttg": result.current_ttg,
            "used_this_week": result.used_this_week,
            "last_week_days": last_week_days,
            "total_refines": result.total_refines,
            "total_cost": result.total_cost,
            "achieved_probability": result.achieved_probability,
        }

        for i, refs in enumerate(result.week_refines, start=1):
            row[f"week_{i}"] = refs

        rows.append(row)

    return rows

def even_split_evaluation(
    target_ttg: int,
    total_refines: int,
    weeks: int,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    front_load_remainder: bool = True,
) -> dict:
    plan = even_split_plan(
        total_refines,
        weeks,
        used_this_week,
        front_load_remainder=front_load_remainder,
    )
    return evaluate_plan(
        target_ttg=target_ttg,
        week_refines=plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
    )


def all_in_evaluation(
    target_ttg: int,
    weeks: int,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
) -> dict:
    plan = all_in_plan(weeks, used_this_week)
    return evaluate_plan(
        target_ttg=target_ttg,
        week_refines=plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
    )

def compare_plan_set(
    target_ttg: int,
    custom_plan: Iterable[int],
    *,
    weeks: int | None = None,
    desired_probability: float | None = None,
    current_ttg: int = 0,
    used_this_week: int = 0,
    last_week_days: int = 5,
    include_optimizer: bool = True,
) -> list[dict]:
    custom_plan = tuple(int(x) for x in custom_plan)
    last_week_days = normalize_last_week_days(last_week_days)

    if weeks is None:
        weeks = len(custom_plan)

    rows = []

    custom_eval = evaluate_plan(
        target_ttg=target_ttg,
        week_refines=custom_plan,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )
    rows.append({"plan_type": "custom", **custom_eval})

    even_plan = even_split_plan(
        sum(custom_plan),
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
    rows.append({"plan_type": "even_split_same_total", **even_eval})

    all_in = all_in_plan(weeks, used_this_week)
    all_in_eval = evaluate_plan(
        target_ttg=target_ttg,
        week_refines=all_in,
        current_ttg=current_ttg,
        used_this_week=used_this_week,
        last_week_days=last_week_days,
    )
    rows.append({"plan_type": "all_in", **all_in_eval})

    if include_optimizer and desired_probability is not None:
        opt = optimal_plan(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=desired_probability,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
            last_week_days=last_week_days,
        )

        rows.append(
            {
                "plan_type": "optimizer",
                "target_ttg": opt.target_ttg,
                "current_ttg": opt.current_ttg,
                "used_this_week": opt.used_this_week,
                "last_week_days": last_week_days,
                "weeks": opt.weeks,
                "week_refines": list(opt.week_refines),
                "total_refines": opt.total_refines,
                "total_cost": opt.total_cost,
                "achieved_probability": opt.achieved_probability,
            }
        )

    return rows

def benchmark_plan_solve(
    target_ttg: int,
    weeks: int,
    desired_probability: float | int | None = None,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
    repeats: int = 1,
) -> dict:
    """
    Time the optimizer for one planning problem.
    """
    requested = normalize_probability_input(desired_probability)

    start = time.perf_counter()
    result = None

    for _ in range(repeats):
        result = optimal_plan(
            target_ttg=target_ttg,
            weeks=weeks,
            desired_probability=requested,
            current_ttg=current_ttg,
            used_this_week=used_this_week,
        )

    elapsed = time.perf_counter() - start

    return {
        "target_ttg": target_ttg,
        "weeks": weeks,
        "desired_probability": requested,
        "current_ttg": current_ttg,
        "used_this_week": used_this_week,
        "repeats": repeats,
        "seconds_total": elapsed,
        "seconds_per_run": elapsed / repeats,
        "result_plan": list(result.week_refines) if result is not None else None,
        "result_cost": result.total_cost if result is not None else None,
        "result_probability": result.achieved_probability if result is not None else None,
    }

def inspect_candidate_family(
    target_ttg: int,
    total_refines: int,
    weeks: int,
    *,
    current_ttg: int = 0,
    used_this_week: int = 0,
) -> list[dict]:
    """
    Evaluate the structured candidate family for one total refine count.
    """
    rows = []

    for plan in _candidate_plans_for_total(
        total_refines=total_refines,
        weeks=weeks,
        used_this_week=used_this_week,
    ):
        rows.append(
            evaluate_plan(
                target_ttg=target_ttg,
                week_refines=plan,
                current_ttg=current_ttg,
                used_this_week=used_this_week,
            )
        )

    return rows