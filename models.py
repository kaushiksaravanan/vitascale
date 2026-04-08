"""Typed Pydantic models for VitaScale environment."""

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional


class Observation(BaseModel):
    """What the agent observes each step."""
    timestamp: int = Field(description="Minutes since episode start")
    current_load: float = Field(description="Requests per minute (0-5000)")
    cpu_util: float = Field(description="CPU utilization 0.0-1.0")
    memory_util: float = Field(description="Memory utilization 0.0-1.0")
    instance_count: int = Field(description="Current number of instances")
    cost_so_far: float = Field(description="Cumulative cost in dollars")
    sla_violation_minutes: int = Field(description="Total SLA violation minutes")
    recent_events: List[str] = Field(default_factory=list, description="Recent failure/event strings")
    difficulty_level: int = Field(description="1=easy, 2=medium, 3=hard")
    pending_requests: float = Field(default=0.0, description="Queued requests not yet served")
    avg_response_time_ms: float = Field(default=50.0, description="Average response time in ms")


class Action(BaseModel):
    """Action the agent takes."""
    action_type: Literal["scale_up", "scale_down", "do_nothing", "migrate_load"] = Field(
        description="Type of scaling action"
    )
    num_instances: int = Field(default=0, ge=0, le=20, description="Number of instances to add/remove")


class Reward(BaseModel):
    """Detailed reward breakdown."""
    total: float = Field(description="Total reward this step")
    efficiency_score: float = Field(default=0.0, description="Reward for cost efficiency")
    sla_score: float = Field(default=0.0, description="Reward for SLA compliance")
    resilience_score: float = Field(default=0.0, description="Reward for handling failures")
    stability_score: float = Field(default=0.0, description="Reward for avoiding thrashing")


class State(BaseModel):
    """Full environment state for checkpointing."""
    task: str = Field(default="")
    step: int = Field(default=0)
    max_steps: int = Field(default=720)
    done: bool = Field(default=True)
    instance_count: int = Field(default=8)
    cost_so_far: float = Field(default=0.0)
    sla_violation_minutes: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class StepResult(BaseModel):
    """Result returned by step() and reset()."""
    observation: Observation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)
