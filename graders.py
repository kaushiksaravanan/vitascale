"""Deterministic graders for VitaScale tasks. Each returns scores in (0, 1) with partial credit."""

from models import Observation


def grade_easy_bench(final_obs: Observation, total_reward: float, history: list) -> float:
    """Grade easy bench: stable diurnal load management.

    Scoring (0.0-1.0):
      - SLA violations: 0.40 weight (0 violations = full, linear decay to 40 min)
      - Cost efficiency: 0.35 weight (under $40 = full, linear decay to $80)
      - Stability: 0.25 weight (fewer scaling actions = better)
    """
    score = 0.0

    # SLA score (40% weight) - 0 violations = perfect, up to 40 min = linear decay
    sla_mins = final_obs.sla_violation_minutes
    if sla_mins == 0:
        score += 0.40
    elif sla_mins <= 40:
        score += 0.40 * (1.0 - sla_mins / 40.0)
    # else: 0

    # Cost score (35% weight) - under $400 = perfect, linear decay to $700
    cost = final_obs.cost_so_far
    if cost <= 400:
        score += 0.35
    elif cost <= 700:
        score += 0.35 * (1.0 - (cost - 400) / 300.0)
    # else: 0

    # Stability score (25% weight) - fewer scaling actions = better
    scale_actions = sum(1 for h in history if h.get("action") in ("scale_up", "scale_down"))
    max_actions = len(history)
    if max_actions > 0:
        action_ratio = scale_actions / max_actions
        # 0-20% actions = full score, linear decay to 80%
        if action_ratio <= 0.20:
            score += 0.25
        elif action_ratio <= 0.80:
            score += 0.25 * (1.0 - (action_ratio - 0.20) / 0.60)

    return round(min(0.999, max(0.001, score)), 4)


def grade_medium_bench(final_obs: Observation, total_reward: float, history: list) -> float:
    """Grade medium bench: diurnal + bursts + node failures.

    Scoring (0.0-1.0):
      - SLA violations: 0.35 weight (0 = perfect, linear to 60 min)
      - Cost efficiency: 0.30 weight (under $50 = perfect, linear to $100)
      - Burst response: 0.20 weight (quick scale-up after load spikes)
      - Recovery: 0.15 weight (recovered from node_down events)
    """
    score = 0.0

    # SLA score (35%)
    sla_mins = final_obs.sla_violation_minutes
    if sla_mins == 0:
        score += 0.35
    elif sla_mins <= 60:
        score += 0.35 * (1.0 - sla_mins / 60.0)

    # Cost score (30%)
    cost = final_obs.cost_so_far
    if cost <= 500:
        score += 0.30
    elif cost <= 900:
        score += 0.30 * (1.0 - (cost - 500) / 400.0)

    # Burst response (20%) - did agent scale up within 3 steps of high load?
    burst_responses = 0
    burst_opportunities = 0
    for i, h in enumerate(history):
        if h.get("load", 0) > 2000:
            burst_opportunities += 1
            # Check next 3 steps for scale_up
            for j in range(i + 1, min(i + 4, len(history))):
                if history[j].get("action") == "scale_up":
                    burst_responses += 1
                    break
    if burst_opportunities > 0:
        score += 0.20 * (burst_responses / burst_opportunities)
    else:
        score += 0.20  # no bursts = full credit

    # Recovery (15%) - instances didn't drop to minimum after failures
    failure_count = sum(1 for h in history if h.get("failure"))
    recovered = sum(1 for i, h in enumerate(history) if h.get("failure") and
                    i + 5 < len(history) and history[i + 5].get("instances", 0) >= 6)
    if failure_count > 0:
        score += 0.15 * (recovered / failure_count)
    else:
        score += 0.15

    return round(min(0.999, max(0.001, score)), 4)


def grade_hard_bench(final_obs: Observation, total_reward: float, history: list) -> float:
    """Grade hard bench: full chaos + curriculum + cost/carbon trade-offs.

    Scoring (0.0-1.0):
      - SLA violations: 0.25 weight (0 = perfect, linear to 90 min)
      - Cost efficiency: 0.25 weight (under $60 = perfect, linear to $130)
      - Failure survival: 0.20 weight (maintained service through failures)
      - Adaptive scaling: 0.15 weight (appropriate instance counts for load)
      - Curriculum response: 0.15 weight (scaled down during easy resets)
    """
    score = 0.0

    # SLA (25%)
    sla_mins = final_obs.sla_violation_minutes
    if sla_mins == 0:
        score += 0.25
    elif sla_mins <= 90:
        score += 0.25 * (1.0 - sla_mins / 90.0)

    # Cost (25%)
    cost = final_obs.cost_so_far
    if cost <= 550:
        score += 0.25
    elif cost <= 1000:
        score += 0.25 * (1.0 - (cost - 550) / 450.0)

    # Failure survival (20%)
    failure_steps = [i for i, h in enumerate(history) if h.get("failure")]
    if failure_steps:
        survived = 0
        for fi in failure_steps:
            # Check if SLA held for 5 steps after failure
            violations_after = sum(
                1 for j in range(fi, min(fi + 5, len(history)))
                if history[j].get("sla_violated", False)
            )
            if violations_after <= 1:
                survived += 1
        score += 0.20 * (survived / len(failure_steps))
    else:
        score += 0.20

    # Adaptive scaling (15%) - instance count roughly proportional to load
    good_scaling = 0
    for h in history:
        load = h.get("load", 0)
        instances = h.get("instances", 8)
        ideal = max(2, min(30, int(load / 160)))
        if abs(instances - ideal) <= 3:
            good_scaling += 1
    if history:
        score += 0.15 * (good_scaling / len(history))

    # Curriculum response (15%) - scaled down during easy_reset periods
    curriculum_steps = [i for i, h in enumerate(history) if h.get("curriculum")]
    if curriculum_steps:
        efficient = sum(1 for ci in curriculum_steps
                       if ci < len(history) and history[ci].get("instances", 30) <= 10)
        score += 0.15 * (efficient / len(curriculum_steps))
    else:
        score += 0.15

    return round(min(0.999, max(0.001, score)), 4)


GRADERS = {
    "easy_bench": grade_easy_bench,
    "medium_bench": grade_medium_bench,
    "hard_bench": grade_hard_bench,
}
