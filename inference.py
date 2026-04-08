"""Baseline inference script for VitaScale environment.

Uses an LLM via OpenAI API to decide scaling actions for cloud resource management.

Emits structured [START], [STEP], [END] logs for evaluation.

Environment variables:
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - Model identifier
    HF_TOKEN      - API authentication token
    ENV_URL       - Environment server URL (default: http://localhost:7860)
"""

import os
import json
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

BENCHMARK = "vitascale"
MAX_STEPS = 720
SUCCESS_SCORE_THRESHOLD = 0.65
TASKS = ["easy_bench", "medium_bench", "hard_bench"]
LLM_CALL_INTERVAL = 5  # Call LLM every N steps, rule-based in between (stay under 20min)


# ── Structured Logging ─────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type": "[START]",
        "task": task,
        "env": env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(json.dumps({
        "type": "[STEP]",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "type": "[END]",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }), flush=True)


# ── LLM Integration ───────────────────────────────────────────
def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    )


SYSTEM_PROMPT = """You are a cloud infrastructure autoscaler. You manage a fleet of compute instances serving web traffic.

You receive the current state: load (requests/min), CPU utilization, instance count, cost, SLA violations, and recent events.

You must respond with EXACTLY one JSON object (no markdown, no explanation):
{"action_type": "<scale_up|scale_down|do_nothing|migrate_load>", "num_instances": <0-20>}

Strategy guidelines:
- Each instance handles ~175 requests/min
- Keep 10-20% headroom above current load
- Scale down when load drops to save cost
- React quickly to spikes (scale_up with enough instances)
- After node_down events, scale_up to replace lost capacity
- During low-load periods, minimize instances (but keep >= 3)
- Avoid thrashing: don't scale up and down every step"""


def get_scaling_action(
    client: OpenAI,
    obs: dict,
    history: Optional[List[str]] = None,
) -> dict:
    """Ask the LLM for a scaling decision."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_msg = (
        f"Current state:\n"
        f"- Load: {obs['current_load']:.0f} req/min\n"
        f"- CPU: {obs['cpu_util']:.1%}\n"
        f"- Instances: {obs['instance_count']}\n"
        f"- Cost so far: ${obs['cost_so_far']:.2f}\n"
        f"- SLA violations: {obs['sla_violation_minutes']} min\n"
        f"- Pending requests: {obs.get('pending_requests', 0):.0f}\n"
        f"- Response time: {obs.get('avg_response_time_ms', 50):.0f}ms\n"
        f"- Recent events: {obs['recent_events']}\n"
        f"- Step: {obs['timestamp']}/{MAX_STEPS}\n"
    )

    if history:
        user_msg += "\nRecent actions:\n" + "\n".join(history[-5:])

    user_msg += "\n\nDecide the next action. Respond with JSON only."
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        # Clean markdown
        if text.startswith("```"):
            lines = text.split("\n")
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end_idx = i
                    break
            text = "\n".join(lines[1:end_idx])
        return json.loads(text)
    except Exception:
        # Fallback: simple rule-based
        return _rule_based_action(obs)


def _rule_based_action(obs: dict) -> dict:
    """Simple rule-based fallback when LLM fails."""
    load = obs["current_load"]
    instances = obs["instance_count"]
    capacity = instances * 175

    if load > capacity * 0.85:
        needed = max(1, int((load - capacity * 0.7) / 175))
        return {"action_type": "scale_up", "num_instances": min(needed, 5)}
    elif load < capacity * 0.4 and instances > 3:
        excess = max(1, int((capacity * 0.4 - load) / 175))
        return {"action_type": "scale_down", "num_instances": min(excess, 3)}
    return {"action_type": "do_nothing", "num_instances": 0}


# ── Environment Client ─────────────────────────────────────────
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def reset(self, task_id: str) -> dict:
        resp = self.client.post(f"{self.base_url}/reset", params={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str, num_instances: int) -> dict:
        resp = self.client.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "num_instances": num_instances},
        )
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ── Main Inference Loop ────────────────────────────────────────
def run_task(task_id: str, llm_client: OpenAI, env_client: EnvClient) -> dict:
    """Run inference on a single task."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_client.reset(task_id)
        obs = result["observation"]

        for step_num in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            # Call LLM every N steps to stay under 20min runtime; rule-based otherwise
            if step_num % LLM_CALL_INTERVAL == 1:
                action_dict = get_scaling_action(
                    client=llm_client,
                    obs=obs,
                    history=history,
                )
            else:
                action_dict = _rule_based_action(obs)

            action_type = action_dict.get("action_type", "do_nothing")
            num_instances = action_dict.get("num_instances", 0)

            result = env_client.step(action_type, num_instances)
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=f"{action_type}({num_instances})",
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step_num}: {action_type}({num_instances}) "
                f"load={obs['current_load']:.0f} inst={obs['instance_count']} "
                f"reward={reward:+.3f}"
            )

            if done:
                break

        # Get final score from grader
        if result.get("info", {}).get("final_score") is not None:
            score = result["info"]["final_score"]
        else:
            score = sum(rewards) / (MAX_STEPS * 0.35) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error running task {task_id}: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task": task_id, "score": score, "success": success, "steps": steps_taken}


def main():
    """Run baseline inference on all tasks."""
    llm_client = get_llm_client()
    env_client = EnvClient(ENV_URL)

    results = []
    try:
        for task_id in TASKS:
            task_result = run_task(task_id, llm_client, env_client)
            results.append(task_result)
            print(
                f"\n[RESULT] Task: {task_result['task']}, Score: {task_result['score']:.2f}, "
                f"Success: {task_result['success']}, Steps: {task_result['steps']}",
                flush=True,
            )
    finally:
        env_client.close()

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n[SUMMARY] Average score: {avg_score:.2f}", flush=True)
    for r in results:
        print(f"  {r['task']}: {r['score']:.2f} ({'PASS' if r['success'] else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
