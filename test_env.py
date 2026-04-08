"""Test VitaScale environment logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import VitaScaleEnv
from models import Action
from graders import GRADERS

env = VitaScaleEnv()

# ── Test all 3 tasks ──
for task_id in ["easy_bench", "medium_bench", "hard_bench"]:
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")

    result = env.reset(task=task_id)
    print(f"Reset OK. obs.load={result.observation.current_load:.0f}, "
          f"instances={result.observation.instance_count}, done={result.done}")

    total_reward = 0.0
    steps = 0
    for step in range(720):
        if result.done:
            break
        # Simple rule-based agent
        load = result.observation.current_load
        instances = result.observation.instance_count
        capacity = instances * 175

        if load > capacity * 0.80:
            needed = max(1, min(5, int((load - capacity * 0.7) / 175)))
            action = Action(action_type="scale_up", num_instances=needed)
        elif load < capacity * 0.45 and instances > 3:
            excess = max(1, min(3, int((capacity * 0.45 - load) / 175)))
            action = Action(action_type="scale_down", num_instances=excess)
        else:
            action = Action(action_type="do_nothing", num_instances=0)

        result = env.step(action)
        total_reward += result.reward
        steps = step + 1

    obs = result.observation
    print(f"Done after {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Cost: ${obs.cost_so_far:.2f}")
    print(f"  SLA violations: {obs.sla_violation_minutes} min")
    print(f"  Final instances: {obs.instance_count}")

    # Run grader
    grader = GRADERS[task_id]
    history = env.get_history()
    score = grader(obs, total_reward, history)
    print(f"  GRADER SCORE: {score:.4f}")

# ── Test state endpoint ──
state = env.state()
print(f"\nState: task={state.task}, step={state.step}, done={state.done}")

# ── Test error handling ──
try:
    env2 = VitaScaleEnv()
    env2.step(Action(action_type="do_nothing", num_instances=0))
except RuntimeError as e:
    print(f"\nExpected error: {e}")

# ── Reproducibility check ──
env3 = VitaScaleEnv()
r1 = env3.reset(task="easy_bench")
r1_step = env3.step(Action(action_type="do_nothing", num_instances=0))

env4 = VitaScaleEnv()
r2 = env4.reset(task="easy_bench")
r2_step = env4.step(Action(action_type="do_nothing", num_instances=0))

assert r1.observation.current_load == r2.observation.current_load, "Traces not deterministic!"
assert r1_step.reward == r2_step.reward, "Rewards not deterministic!"
print("\nDeterminism check PASSED")

print("\nAll tests passed!")
