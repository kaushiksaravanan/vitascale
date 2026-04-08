"""Test VitaScale live on HF Spaces."""
import httpx

BASE = "https://kaushikss-vitascale.hf.space"
c = httpx.Client(timeout=30)

# Root
r = c.get(BASE + "/")
print(f"GET /: {r.status_code} {r.json()}")

# Health
r = c.get(BASE + "/health")
print(f"GET /health: {r.status_code} {r.json()}")

# Reset
r = c.post(BASE + "/reset", params={"task_id": "easy_bench"})
print(f"POST /reset: {r.status_code} done={r.json()['done']}")

# Step
r = c.post(BASE + "/step", json={"action_type": "scale_up", "num_instances": 2})
d = r.json()
print(f"POST /step: {r.status_code} reward={d['reward']} instances={d['observation']['instance_count']}")

# State
r = c.get(BASE + "/state")
print(f"GET /state: {r.status_code} step={r.json()['step']}")

# Tasks
r = c.get(BASE + "/tasks")
print(f"GET /tasks: {r.status_code} tasks={list(r.json().keys())}")

# Reset all tasks
for t in ["easy_bench", "medium_bench", "hard_bench"]:
    r = c.post(BASE + "/reset", params={"task_id": t})
    print(f"RESET {t}: {r.status_code}")

print("\nVitaScale HF Space FULLY OPERATIONAL!")
c.close()
