"""Core VitaScale environment implementing step/reset/state."""

import math
from typing import Optional, Dict, Any, List

from models import Observation, Action, State, StepResult
from load_traces import LOAD_TRACES, CURRICULUM_INJECTIONS
from graders import GRADERS


class VitaScaleEnv:
    """Cloud resource orchestration environment.

    The agent manages a fleet of cloud instances to serve fluctuating load
    while minimizing cost and SLA violations.

    Reward design (per step):
      - Base: +0.10 for running the system
      - SLA compliance: +0.25 if no violation this step, -0.40 if violated
      - Cost efficiency: 0.0 to -0.20 based on over-provisioning
      - Stability: -0.05 per scaling action (penalizes thrashing)
      - Failure handling: +0.15 bonus if survived a failure event
      - Curriculum: +0.10 during easy_reset if instances < 10
    """

    CAPACITY_PER_INSTANCE = 175  # requests/min each instance can handle

    def __init__(self):
        self._task: Optional[str] = None
        self._trace: List[Dict] = []
        self._step: int = 0
        self._max_steps: int = 720
        self._instance_count: int = 8
        self._cost_so_far: float = 0.0
        self._sla_violation_minutes: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._history: List[Dict[str, Any]] = []
        self._prev_action: Optional[str] = None
        self._pending_requests: float = 0.0

    def reset(self, task: Optional[str] = None) -> StepResult:
        if task is None:
            task = "easy_bench"
        if task not in LOAD_TRACES:
            raise ValueError(f"Unknown task: {task}. Available: {list(LOAD_TRACES.keys())}")

        self._task = task
        self._trace = LOAD_TRACES[task]
        self._step = 0
        self._max_steps = 720
        self._instance_count = 8
        self._cost_so_far = 0.0
        self._sla_violation_minutes = 0
        self._total_reward = 0.0
        self._done = False
        self._history = []
        self._prev_action = None
        self._pending_requests = 0.0

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"task": task, "difficulty": self._difficulty()},
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Apply action
        old_instances = self._instance_count
        if action.action_type == "scale_up":
            self._instance_count = min(30, self._instance_count + max(1, action.num_instances))
        elif action.action_type == "scale_down":
            self._instance_count = max(2, self._instance_count - max(1, action.num_instances))
        elif action.action_type == "migrate_load":
            # Rebalance: reduces pending requests but costs a step
            self._pending_requests = max(0, self._pending_requests - self._instance_count * 50)

        # Get trace data
        data = self._trace[self._step % len(self._trace)]
        load = data["load"] + data.get("spike", 0)
        failure = data.get("failure")

        # Check curriculum injection (hard bench only)
        in_curriculum = False
        if self._task == "hard_bench":
            for inj in CURRICULUM_INJECTIONS:
                if inj["start_minute"] <= self._step <= inj["start_minute"] + inj["duration"]:
                    load = 600  # easy reset
                    in_curriculum = True
                    break

        # Handle failures
        if failure == "node_down":
            self._instance_count = max(2, self._instance_count - 1)
        elif failure == "cascade_failure":
            self._instance_count = max(2, self._instance_count - 2)

        # Compute capacity and SLA
        capacity = self._instance_count * self.CAPACITY_PER_INSTANCE
        sla_violated = load > capacity
        if sla_violated:
            self._sla_violation_minutes += 1
            self._pending_requests += load - capacity

        # Pending requests drain over time
        serve_rate = max(0, capacity - load)
        self._pending_requests = max(0, self._pending_requests - serve_rate * 0.3)

        # Cost: $0.10/instance/step, price spikes double cost
        price_multiplier = 2.0 if failure == "price_spike" else 1.0
        step_cost = self._instance_count * 0.10 * price_multiplier
        self._cost_so_far += step_cost

        # Response time simulation
        utilization = min(0.99, load / capacity) if capacity > 0 else 0.99
        avg_response_ms = 20 + 180 * (utilization ** 3) + (self._pending_requests / 100)

        # ── Reward calculation ──
        # Base reward
        reward = 0.10

        # SLA compliance
        if not sla_violated:
            reward += 0.25
        else:
            overload_ratio = min(1.0, (load - capacity) / capacity) if capacity > 0 else 1.0
            reward -= 0.40 * overload_ratio

        # Cost efficiency: penalize over-provisioning
        ideal_instances = max(2, math.ceil(load / self.CAPACITY_PER_INSTANCE))
        excess = max(0, self._instance_count - ideal_instances - 2)  # 2 buffer ok
        reward -= 0.03 * excess

        # Stability: penalize thrashing
        if action.action_type in ("scale_up", "scale_down"):
            reward -= 0.05

        # Failure handling bonus
        if failure and not sla_violated:
            reward += 0.15

        # Curriculum response (hard bench)
        if in_curriculum and self._instance_count <= 10:
            reward += 0.10

        self._total_reward += reward

        # Record history
        self._history.append({
            "step": self._step,
            "action": action.action_type,
            "num_instances": action.num_instances,
            "instances": self._instance_count,
            "load": round(load, 1),
            "cost": round(step_cost, 4),
            "sla_violated": sla_violated,
            "failure": failure,
            "reward": round(reward, 4),
            "curriculum": in_curriculum,
        })

        self._step += 1
        self._prev_action = action.action_type
        self._done = self._step >= self._max_steps

        obs = self._make_observation()
        info: Dict[str, Any] = {
            "step": self._step,
            "reward_breakdown": {
                "total": round(reward, 4),
                "sla_violated": sla_violated,
                "failure": failure,
                "in_curriculum": in_curriculum,
            },
        }

        # If done, run grader
        if self._done:
            grader = GRADERS.get(self._task)
            if grader:
                final_score = grader(obs, self._total_reward, self._history)
                info["final_score"] = final_score
                info["grader_task"] = self._task

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        return State(
            task=self._task or "",
            step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            instance_count=self._instance_count,
            cost_so_far=round(self._cost_so_far, 2),
            sla_violation_minutes=self._sla_violation_minutes,
            total_reward=round(self._total_reward, 4),
            history=self._history[-10:],  # last 10 steps
        )

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history

    def _make_observation(self) -> Observation:
        data = self._trace[self._step % len(self._trace)]
        load = data["load"] + data.get("spike", 0)
        capacity = self._instance_count * self.CAPACITY_PER_INSTANCE
        cpu = min(0.99, load / capacity) if capacity > 0 else 0.99
        avg_resp = 20 + 180 * (cpu ** 3) + (self._pending_requests / 100)

        recent = []
        for h in self._history[-5:]:
            if h.get("failure"):
                recent.append(h["failure"])

        return Observation(
            timestamp=self._step,
            current_load=round(load, 1),
            cpu_util=round(cpu, 3),
            memory_util=round(cpu * 0.82, 3),
            instance_count=self._instance_count,
            cost_so_far=round(self._cost_so_far, 2),
            sla_violation_minutes=self._sla_violation_minutes,
            recent_events=recent,
            difficulty_level=self._difficulty(),
            pending_requests=round(self._pending_requests, 1),
            avg_response_time_ms=round(avg_resp, 1),
        )

    def _difficulty(self) -> int:
        if self._task == "hard_bench":
            return 3
        elif self._task == "medium_bench":
            return 2
        return 1
