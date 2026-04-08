"""Baseline inference script for VitaScale environment."""

import os
import sys
import json
import httpx
from openai import OpenAI

# ── Environment variables (checklist-compliant) ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "https://kaushikss-vitascale.hf.space")
LLM_CALL_INTERVAL = 5

# ── OpenAI client ────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Environment client ───────────────────────────────────────────────
class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60)

    def reset(self, task_id: str) -> dict:
        r = self.http.post(f"{self.base}/reset", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.http.post(f"{self.base}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.http.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()


# ── Rule-based fallback policy ───────────────────────────────────────
def rule_based_action(obs: dict) -> dict:
    load = obs.get("current_load", 0)
    instances = obs.get("instance_count", 1)
    cpu = obs.get("cpu_util", 0)
    capacity = instances * 175

    if load > capacity * 0.85 or cpu > 0.80:
        add = max(1, int((load - capacity * 0.7) / 175))
        add = min(add, 10)
        return {"action_type": "scale_up", "num_instances": add}
    elif load < capacity * 0.35 and instances > 2 and cpu < 0.30:
        remove = max(1, min(int((capacity * 0.35 - load) / 175), instances - 2))
        return {"action_type": "scale_down", "num_instances": remove}
    else:
        return {"action_type": "do_nothing", "num_instances": 0}


# ── LLM-based policy ────────────────────────────────────────────────
def llm_action(obs: dict, step_num: int) -> dict:
    prompt = (
        f"You are a cloud autoscaling agent managing a production cluster.\n"
        f"Current state at step {step_num}:\n"
        f"  - Load: {obs.get('current_load', 0):.0f} req/min\n"
        f"  - Instances: {obs.get('instance_count', 1)}\n"
        f"  - CPU: {obs.get('cpu_util', 0):.1%}\n"
        f"  - Memory: {obs.get('memory_util', 0):.1%}\n"
        f"  - Cost so far: ${obs.get('cost_so_far', 0):.2f}\n"
        f"  - SLA violations: {obs.get('sla_violation_minutes', 0)} min\n"
        f"  - Pending requests: {obs.get('pending_requests', 0):.0f}\n"
        f"  - Avg response time: {obs.get('avg_response_time_ms', 50):.0f} ms\n"
        f"  - Recent events: {obs.get('recent_events', [])}\n"
        f"  - Capacity per instance: 175 req/min\n\n"
        f"Respond with EXACTLY one JSON object:\n"
        f'{{"action_type": "<scale_up|scale_down|do_nothing>", "num_instances": <int>}}\n'
        f"No markdown, no explanation."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        action = json.loads(text)
        if action.get("action_type") not in ("scale_up", "scale_down", "do_nothing", "migrate_load"):
            return rule_based_action(obs)
        action["num_instances"] = max(0, min(20, int(action.get("num_instances", 0))))
        return action
    except Exception:
        return rule_based_action(obs)


# ── Run one task ─────────────────────────────────────────────────────
def run_task(env: EnvClient, task_id: str):
    result = env.reset(task_id)
    obs = result["observation"]
    total_reward = 0.0
    step_num = 0

    print(f"[START] task={task_id}")

    while not result.get("done", False):
        if step_num % LLM_CALL_INTERVAL == 0:
            action = llm_action(obs, step_num)
        else:
            action = rule_based_action(obs)

        result = env.step(action)
        obs = result["observation"]
        reward = result.get("reward", 0)
        total_reward += reward
        step_num += 1

        print(f"[STEP] step={step_num} reward={reward:.4f} total={total_reward:.4f} action={action['action_type']} instances={obs.get('instance_count', 0)}")

    print(f"[END] task={task_id} steps={step_num} total_reward={total_reward:.4f}")
    return total_reward


# ── Main ─────────────────────────────────────────────────────────────
def main():
    env = EnvClient(ENV_URL)
    tasks = ["easy_bench", "medium_bench", "hard_bench"]
    results = {}
    for task_id in tasks:
        score = run_task(env, task_id)
        results[task_id] = score
    print(f"\n=== Summary ===")
    for tid, sc in results.items():
        print(f"  {tid}: {sc:.4f}")


if __name__ == "__main__":
    main()
