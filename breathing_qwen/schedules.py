from __future__ import annotations

from dataclasses import dataclass

_VALID_PHASES = {"open", "close"}


@dataclass(frozen=True)
class BreathingSchedule:
    betas: tuple[float, ...]
    phases: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.betas) != len(self.phases):
            raise ValueError("betas and phases must have the same length")
        if not self.betas:
            raise ValueError("schedule must contain at least one cycle")
        if any(beta <= 0 for beta in self.betas):
            raise ValueError("all beta values must be positive")
        unknown = [phase for phase in self.phases if phase not in _VALID_PHASES]
        if unknown:
            raise ValueError(f"unknown phase: {unknown[0]}")

    def __len__(self) -> int:
        return len(self.betas)


DEFAULT_SCHEDULE = BreathingSchedule(
    betas=(0.50, 1.75, 0.65, 2.00),
    phases=("open", "close", "open", "close"),
)
