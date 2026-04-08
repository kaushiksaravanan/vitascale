"""Deterministic load traces for VitaScale environment.

All traces are pre-computed with fixed seeds for reproducibility.
Each trace is a list of dicts with minute-by-minute data for 1440 minutes (24h).
"""

import random
import math
from typing import List, Dict, Optional

def _diurnal(minute: int) -> float:
    """Realistic diurnal pattern: low at night, peak at 2pm, secondary peak at 10am."""
    hour = (minute % 1440) / 60.0
    # Primary peak at 14:00, secondary at 10:00, trough at 4:00
    primary = math.exp(-0.5 * ((hour - 14) / 3) ** 2) * 1400
    secondary = math.exp(-0.5 * ((hour - 10) / 2.5) ** 2) * 800
    base = 400 + primary + secondary
    return base


def generate_easy_trace() -> List[Dict]:
    """Stable diurnal load. No spikes, no failures. Predictable."""
    rng = random.Random(42)
    trace = []
    for i in range(1440):
        base = _diurnal(i)
        noise = rng.gauss(0, 30)  # small noise
        trace.append({
            "minute": i,
            "load": round(max(200, base + noise), 1),
            "spike": 0.0,
            "failure": None,
        })
    return trace


def generate_medium_trace() -> List[Dict]:
    """Diurnal + 6 random traffic bursts + 2 node failures."""
    rng = random.Random(123)
    # Pre-define burst minutes for reproducibility
    burst_minutes = {187, 412, 633, 891, 1045, 1287}
    failure_minutes = {420, 980}
    trace = []
    for i in range(1440):
        base = _diurnal(i)
        noise = rng.gauss(0, 50)
        spike = 0.0
        if i in burst_minutes:
            spike = rng.uniform(800, 1500)
        # Burst aftermath (elevated for 15 min after burst)
        for bm in burst_minutes:
            if bm < i <= bm + 15:
                spike += max(0, (bm + 15 - i) / 15 * 600)
        failure = None
        if i in failure_minutes:
            failure = "node_down"
        trace.append({
            "minute": i,
            "load": round(max(200, base + noise + spike), 1),
            "spike": round(spike, 1),
            "failure": failure,
        })
    return trace


def generate_hard_trace() -> List[Dict]:
    """Full chaos: noisy load, frequent bursts, cascading failures, price/carbon spikes."""
    rng = random.Random(777)
    # Pre-defined events for reproducibility
    burst_minutes = {95, 218, 337, 445, 589, 672, 788, 901, 1023, 1156, 1289, 1380}
    events = {
        150: "node_down",
        345: "node_down",
        520: "price_spike",
        690: "node_down",
        780: "carbon_peak",
        920: "cascade_failure",
        1050: "price_spike",
        1200: "node_down",
        1340: "carbon_peak",
    }
    trace = []
    for i in range(1440):
        base = _diurnal(i)
        noise = rng.gauss(0, 120)  # high noise
        spike = 0.0
        if i in burst_minutes:
            spike = rng.uniform(1200, 2200)
        for bm in burst_minutes:
            if bm < i <= bm + 20:
                spike += max(0, (bm + 20 - i) / 20 * 800)
        failure = events.get(i)
        # Cascade failure causes elevated load for 30 min
        if any(events.get(m) == "cascade_failure" and m < i <= m + 30 for m in events):
            spike += 500
        trace.append({
            "minute": i,
            "load": round(max(200, base + noise + spike), 1),
            "spike": round(spike, 1),
            "failure": failure,
        })
    return trace


# Pre-compute all traces (deterministic, same every run)
LOAD_TRACES = {
    "easy_bench": generate_easy_trace(),
    "medium_bench": generate_medium_trace(),
    "hard_bench": generate_hard_trace(),
}

# Curriculum injection schedule for hard bench
CURRICULUM_INJECTIONS = [
    {"start_minute": 180, "duration": 25, "type": "easy_reset"},
    {"start_minute": 420, "duration": 30, "type": "easy_reset"},
    {"start_minute": 780, "duration": 20, "type": "easy_reset"},
    {"start_minute": 1100, "duration": 35, "type": "easy_reset"},
]
