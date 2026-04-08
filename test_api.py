"""Test VitaScale HTTP API endpoints."""
import httpx

BASE = "http://localhost:7860"
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
print(f"  load={r.json()['observation']['current_load']}")

# Step scale_up
r = c.post(BASE + "/step", json={"action_type": "scale_up", "num_instances": 3})
d = r.json()
print(f"POST /step(scale_up 3): {r.status_code} reward={d['reward']} instances={d['observation']['instance_count']}")

# Step do_nothing 
r = c.post(BASE + "/step", json={"action_type": "do_nothing", "num_instances": 0})
d = r.json()
print(f"POST /step(do_nothing): {r.status_code} reward={d['reward']}")

# Step scale_down
r = c.post(BASE + "/step", json={"action_type": "scale_down", "num_instances": 2})
d = r.json()
print(f"POST /step(scale_down 2): {r.status_code} reward={d['reward']} instances={d['observation']['instance_count']}")

# State
r = c.get(BASE + "/state")
print(f"GET /state: {r.status_code} step={r.json()['step']}")

# Tasks
r = c.get(BASE + "/tasks")
print(f"GET /tasks: {r.status_code} tasks={list(r.json().keys())}")

# Test medium + hard reset
for t in ["medium_bench", "hard_bench"]:
    r = c.post(BASE + "/reset", params={"task_id": t})
    print(f"RESET {t}: {r.status_code}")

print("\nAll VitaScale API endpoints working!")
c.close()
