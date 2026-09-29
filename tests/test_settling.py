import numpy as np
import pytest

from breathing_qwen.schedules import DEFAULT_SCHEDULE
from breathing_qwen.settling import settle


EVIDENCE = np.array(
    [
        [3.0, 0.0, -1.0],
        [2.5, 0.2, -0.5],
        [2.8, 0.1, -0.7],
        [-2.0, 5.0, 0.0],
    ],
    dtype=float,
)


def test_one_shot_has_one_cycle_and_never_changes_trust():
    trace = settle(EVIDENCE, DEFAULT_SCHEDULE, mode="one_shot", delta=1.5, weight_floor=0.05)
    assert len(trace.cycles) == 1
    assert np.allclose(trace.cycles[0].cue_weights, 1.0)


def test_fixed_iteration_keeps_trust_frozen():
    trace = settle(EVIDENCE, DEFAULT_SCHEDULE, mode="fixed", delta=1.5, weight_floor=0.05)
    assert len(trace.cycles) == len(DEFAULT_SCHEDULE)
    assert all(np.allclose(c.cue_weights, 1.0) for c in trace.cycles)
    assert all(c.beta == pytest.approx(1.0) for c in trace.cycles)


def test_breathing_only_follows_beta_schedule_but_keeps_trust_frozen():
    trace = settle(EVIDENCE, DEFAULT_SCHEDULE, mode="breathing", delta=1.5, weight_floor=0.05)
    assert [c.beta for c in trace.cycles] == list(DEFAULT_SCHEDULE.betas)
    assert all(np.allclose(c.cue_weights, 1.0) for c in trace.cycles)
    assert trace.cycles[0].entropy > trace.cycles[1].entropy


def test_breathing_residue_downweights_inconsistent_cue():
    trace = settle(EVIDENCE, DEFAULT_SCHEDULE, mode="breathing_residue", delta=1.0, weight_floor=0.05)
    initial = trace.cycles[0].cue_weights
    final = trace.cycles[-1].cue_weights
    assert np.allclose(initial, 1.0)
    assert final[3] < final[:3].min()
    assert final[3] >= 0.05
    assert trace.winner_index == 0


def test_external_settling_is_cue_order_invariant():
    original = settle(EVIDENCE, DEFAULT_SCHEDULE, mode="breathing_residue", delta=1.0, weight_floor=0.05)
    perm = np.array([3, 0, 2, 1])
    shuffled = settle(EVIDENCE[perm], DEFAULT_SCHEDULE, mode="breathing_residue", delta=1.0, weight_floor=0.05)
    assert np.allclose(original.final_probabilities, shuffled.final_probabilities)
    inverse = np.argsort(perm)
    assert np.allclose(original.cycles[-1].cue_weights, shuffled.cycles[-1].cue_weights[inverse])


def test_settle_rejects_nonfinite_evidence():
    bad = EVIDENCE.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        settle(bad, DEFAULT_SCHEDULE, mode="fixed", delta=1.5, weight_floor=0.05)


def test_settle_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        settle(EVIDENCE, DEFAULT_SCHEDULE, mode="oracle", delta=1.5, weight_floor=0.05)
