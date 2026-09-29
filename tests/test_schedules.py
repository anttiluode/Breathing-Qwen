import pytest

from breathing_qwen.schedules import BreathingSchedule, DEFAULT_SCHEDULE


def test_default_schedule_is_frozen():
    assert DEFAULT_SCHEDULE.betas == (0.50, 1.75, 0.65, 2.00)
    assert DEFAULT_SCHEDULE.phases == ("open", "close", "open", "close")


def test_schedule_rejects_non_positive_beta():
    with pytest.raises(ValueError, match="positive"):
        BreathingSchedule((0.5, 0.0), ("open", "close"))


def test_schedule_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        BreathingSchedule((0.5, 1.5), ("open",))


def test_schedule_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        BreathingSchedule((0.5,), ("blur",))
