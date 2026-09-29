from __future__ import annotations

from dataclasses import dataclass

from .schedules import BreathingSchedule, DEFAULT_SCHEDULE


@dataclass(frozen=True)
class SettlingConfig:
    schedule: BreathingSchedule = DEFAULT_SCHEDULE
    fixed_beta: float = 1.0
    huber_delta: float = 1.5
    weight_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.fixed_beta <= 0:
            raise ValueError("fixed_beta must be positive")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0 < self.weight_floor <= 1:
            raise ValueError("weight_floor must be in (0, 1]")
