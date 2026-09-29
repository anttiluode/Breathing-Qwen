# Breathing-Qwen v0 — Design

Date: 2026-09-29  
Status: draft for written-spec review  
Repository: `anttiluode/Breathing-Qwen`

## 1. Purpose

Breathing-Qwen tests one narrow hypothesis:

> A frozen language model can sometimes contain enough information to recover an answer that its ordinary one-pass retrieval misses, and an iterative settling process that alternates broad and sharp retrieval while reducing trust in inconsistent cues can recover that answer without adding new external evidence.

The project is not a claim about brain rhythms, consciousness, or a universal replacement for model scale. The neuroscience analogy motivates the mechanism; the repo tests the mechanism.

The initial target is Qwen3-8B because the existing AdaptiveObserverCache work already established a practical local setup for frozen Qwen3-8B, KV-cache inspection, source spans, and constrained-memory execution.

## 2. Core distinction

The repo must keep three ideas separate.

1. **Generation temperature** controls randomness while sampling output tokens. We do not use this as the intervention. Evaluation is deterministic.
2. **Retrieval temperature** (inverse temperature `beta`) changes the sharpness of candidate or attention competition before an answer is emitted.
3. **Residue-dependent trust** changes which cues or source spans are allowed to influence later retrieval cycles.

The scientific comparison is therefore not "hot Qwen versus cold Qwen." It is fixed evidence and fixed weights under different inference dynamics.

## 3. Hypotheses

### H0 — breathing alone

Oscillating retrieval sharpness without changing trust does not reliably improve corrupted-cue retrieval over a fixed-temperature iterative baseline.

### H1 — residue changes the next computation

When one cue is inconsistent with the rest of a partially specified object, an iterative robust estimator can reduce that cue's authority and recover the correct answer more often than native one-pass retrieval or breathing temperature alone.

### H2 — frozen-model latent recovery

For at least some failures, Qwen3-8B assigns recoverable support to the correct answer before decoding even when the ordinary prompt-level decision is wrong. Iterative settling can expose that support without training the model and without generating reasoning tokens between cycles.

### H3 — internal intervention (stretch gate)

If retrieval-temperature and source-span trust are applied inside selected Qwen attention heads, the model can change its next-token/candidate preference across repeated passes over an unchanged prompt. This is a stronger claim than external settling and must be reported separately.

## 4. Non-goals

v0 will not:

- train or fine-tune Qwen;
- claim that theta rhythm literally implements a softmax temperature;
- claim that a smaller recurrent model can replace an 8B model;
- use chain-of-thought tokens as the settling loop;
- benchmark general reasoning ability;
- hide negative results;
- require a large GPU for CI.

## 5. Experimental ladder

The project is intentionally staged so a failure at a later gate does not invalidate earlier measurements.

### Gate 0 — reference dynamical memory (CPU, no LLM)

Implement a small NumPy associative-retrieval system with stored vectors and corrupted query features.

Four arms:

1. one-shot retrieval;
2. repeated fixed-temperature retrieval;
3. breathing temperature only;
4. breathing temperature plus residue-dependent trust.

The exact same stored memories and corrupted observations are used in every arm. The purpose is to verify that the proposed controller has a regime where it does something distinct and measurable before Qwen complexity is introduced.

A minimal update is:

```text
p_t = softmax(beta_t * score(S, W_t, cues))
residue_t[j] = mismatch(cue_j, retrieved_state_t)
W_{t+1}[j] = robust_weight(residue_t[j])
```

`W` is bounded away from zero so a cue can lose authority without being silently deleted. The final implementation will expose the weighting rule and constants in the receipt.

**Gate 0 passes** if arm 4 improves corrupted-query accuracy over arms 1–3 across a frozen seeded benchmark while not materially degrading clean-query accuracy. The pass threshold is declared in code before the benchmark is run.

### Gate 1 — Qwen external settling (primary v0)

Use frozen Qwen3-8B as a semantic scoring engine. No internal Qwen code is changed for the first real-model gate.

Each benchmark item contains:

- an answer;
- several individually addressable cues describing that answer;
- a clean condition;
- a corrupted condition in which exactly one cue is replaced by a plausible but false cue;
- a small candidate answer set containing the correct answer and hard distractors.

For each cue, Qwen produces a deterministic score vector over the same candidates. These per-cue vectors are the evidence objects. The controller aggregates them iteratively.

A robust settling cycle is:

```text
z_j       = frozen Qwen score vector from cue j
mu_t      = weighted aggregate(z_1 ... z_n; W_t)
p_t       = softmax(beta_t * mu_t)
residue_j = distance(z_j, mu_t)
W_{t+1}   = IRLS-like robust weights(residue)
```

No ground-truth corruption label is supplied to this update. The benchmark label is used only for evaluation after inference.

This gate deliberately uses a model-external controller. If it works, the claim is only that frozen Qwen evidence can be settled more robustly. It is not yet an attention-mechanism result.

### Gate 2 — unchanged-prompt multi-pass Qwen

Run repeated deterministic forward passes over one unchanged prompt. No tokens are generated between passes. Span-level cue weights are carried by the controller from one pass to the next.

The first implementation may express span trust through controlled prompt masking/bias or score aggregation if a stable internal hook is unavailable. Any such mechanism must be named exactly in receipts.

The required observation is a preference trajectory, for example:

```text
cycle 0: distractor > target
cycle 1: distractor ~= target
cycle 2: target > distractor
```

with identical model weights and identical external evidence.

### Gate 3 — internal attention breathing (stretch goal)

Force an attention implementation that exposes pre-softmax attention scores. For a small declared set of layers/heads, replace

```text
softmax(scores)
```

with

```text
softmax(beta_t * scores + span_bias(W_t))
```

All unselected heads remain native. Candidate preference is measured before any answer token is generated.

Head/layer selection must be based on a predeclared causal or span-sensitivity probe, not hand-picked from the final benchmark outcome.

Gate 3 is optional for the first shipped v0. A clean Gate-1 result is more valuable than a fragile internal patch that cannot be reproduced.

## 6. Why this ladder

There are three common confounds this design avoids:

- **sampling confound:** changing output temperature can look like exploration while merely adding randomness;
- **extra-information confound:** generating reasoning tokens changes the context, so a later answer is not inference over the same evidence;
- **controller/model confound:** an external robust aggregator working does not prove Qwen's internal attention implements the same dynamics.

Each gate increases the strength of the claim only after the simpler claim is measured.

## 7. Benchmark

### 7.1 Item schema

Each JSONL item has approximately:

```json
{
  "id": "brain-striatum-001",
  "answer": "striatum",
  "cues": [
    "a brain structure strongly associated with dopamine",
    "involved in interval timing and action selection",
    "part of the basal ganglia",
    "starts with B"
  ],
  "corrupt_index": 3,
  "candidates": ["striatum", "brainstem", "septum", "cerebellum", "basal forebrain"]
}
```

The clean counterpart replaces the corrupted cue with a true one. The `corrupt_index` is hidden from the inference controller and used only in evaluation.

### 7.2 Categories

The first committed benchmark should contain enough variety to prevent one lexical trick from dominating:

- anatomy/science terms;
- common objects;
- people/places with strong semantic associations;
- technical vocabulary;
- category-conflict cases;
- false initial-letter cases;
- false but semantically plausible association cases.

The first version should stay small enough to run locally on Qwen3-8B. A target of 32–64 paired clean/corrupt items is appropriate for v0.

### 7.3 Frozen-before-run rule

Benchmark items, schedules, random seeds, pass thresholds, and arm definitions are committed before the first official Qwen run. Changes after seeing outcomes create a new benchmark version rather than overwriting the old receipt.

## 8. Metrics

Primary metrics:

- candidate accuracy on clean items;
- candidate accuracy on corrupted items;
- corruption penalty: `clean_accuracy - corrupt_accuracy`;
- recovery rate on cases native Qwen gets wrong;
- clean-regression rate;
- mean rank / margin of the correct candidate;
- number of settling cycles and model forward passes.

Mechanism metrics:

- cue trust trajectory by cycle;
- whether the known corrupted cue loses trust (evaluation only);
- candidate probability/margin trajectory;
- entropy trajectory across open/close phases;
- convergence / oscillation / no-decision rate.

Gate 3 additionally records:

- attention mass on each declared cue/source span;
- selected head/layer IDs;
- source K/V integrity hashes when applicable;
- target versus distractor attention trajectory.

## 9. Controls

Every official run includes:

1. native all-cues Qwen;
2. per-cue fixed aggregation, no iteration;
3. repeated fixed-temperature settling;
4. breathing temperature only;
5. breathing plus residue-dependent trust.

Useful ablations:

- reverse breathing schedule;
- shuffled cue order;
- same corrupted cue moved to a different position;
- trust update with `beta` fixed;
- beta breathing with trust frozen;
- random cue down-weighting with matched total weight loss.

The order-swap control is especially important because earlier Qwen observer experiments showed large position/order effects. Breathing-Qwen should not mistake a position correction for a residue mechanism.

## 10. Default breathing schedule

The first schedule is fixed and deliberately simple:

```text
beta = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
```

A fixed-temperature control uses a beta chosen before the official run. Schedule tuning on the official benchmark is prohibited; exploratory tuning uses a separate development split/seed.

The schedule is a hypothesis, not a sacred constant. Receipts always record it.

## 11. Residue and robust weighting

The primary external-settling residue is disagreement between a cue-specific candidate-score vector and the current aggregate candidate-score vector.

Use a robust M-estimator/IRLS-style weighting function with explicit scale normalization. Candidate implementations include Huber or Tukey weights; v0 should choose one and freeze its constant before the benchmark run.

Requirements:

- no answer label enters the update;
- all weights remain inspectable;
- no cue can be deleted silently;
- clean cues that disagree temporarily can recover weight on later cycles if the rule permits it;
- the exact rule is unit-tested on synthetic vectors.

## 12. Qwen scoring

Primary Qwen model: `Qwen/Qwen3-8B`, pinned to an exact model revision in official receipts.

Candidate scoring should prefer teacher-forced conditional log-likelihood over free generation. This makes the comparison deterministic and exposes margins even when greedy generation would emit the same answer in multiple arms.

Thinking/reasoning output is disabled for the core experiment. No chain-of-thought text is needed or stored.

The runtime should support constrained local hardware via `device_map="auto"` and configurable `max_memory`, following the practical pattern already used in the Qwen observer work.

## 13. Package layout

Planned structure:

```text
Breathing-Qwen/
  README.md
  pyproject.toml
  breathing_qwen/
    __init__.py
    config.py
    schedules.py
    robust.py
    toy_memory.py
    qwen_score.py
    settling.py
    benchmark.py
    receipts.py
    internal_attention.py   # Gate 3 only
  benchmarks/
    v0.jsonl
  scripts/
    run_toy.py
    run_qwen.py
    summarize_receipts.py
  tests/
    test_schedules.py
    test_robust.py
    test_toy_memory.py
    test_settling.py
    test_benchmark_schema.py
    test_receipts.py
    test_qwen_smoke.py      # opt-in, no CI model download
  receipts/
    .gitkeep
  docs/
    superpowers/specs/
      2026-09-29-breathing-qwen-v0-design.md
```

## 14. Interfaces

### `BreathingSchedule`

Owns beta-by-cycle and phase labels. Pure data, deterministic, unit-testable.

### `EvidenceScorer`

Interface that converts one cue into a vector of candidate scores. Gate 0 uses a synthetic scorer; Gate 1 uses frozen Qwen.

### `SettlingController`

Consumes cue score vectors, schedule, and robust-weight rule. Produces a cycle-by-cycle trace containing aggregate scores, candidate probabilities, residue, and cue weights.

The controller does not know the correct answer or corrupt cue index.

### `BenchmarkRunner`

Runs every arm on the same frozen items and emits machine-readable receipts.

### `InternalAttentionAdapter`

Optional Gate-3 component. Owns all model-specific hooking so the core settling code remains independent of Transformers internals.

## 15. Error handling

- Missing model or insufficient memory: fail with a clear message and preserve any completed per-item receipts.
- NaN/Inf scores: mark the item invalid and fail the official run rather than silently clipping.
- Candidate-tokenization edge cases: score full answer strings, length-normalized, and record token counts.
- Non-convergence: record explicitly; do not force a winner.
- Unsupported Transformers/Qwen internals for Gate 3: Gate 1/2 remain runnable; Gate 3 exits with a versioned compatibility error.

## 16. Testing strategy

CI must never download Qwen3-8B.

Unit tests cover:

- schedule determinism;
- monotonic properties of robust weights;
- residue invariants;
- controller blindness to benchmark labels;
- clean/corrupt schema pairing;
- deterministic seeded Gate-0 receipts;
- JSON receipt round-trip;
- CLI argument validation.

An opt-in integration test (`pytest -m qwen`) runs only when the model is available locally or explicit download permission is supplied.

Gate 0 is the CI scientific smoke test: the controller must reproduce a frozen synthetic receipt exactly within tolerances.

## 17. Receipts and provenance

Every run writes a timestamped JSON receipt containing:

- git commit SHA;
- Python, PyTorch, Transformers versions;
- model name and exact revision;
- device and memory configuration;
- benchmark version/hash;
- random seeds;
- arm configuration;
- breathing schedule;
- robust weighting rule/constants;
- aggregate metrics;
- per-item predictions and cycle traces;
- declared gate thresholds and pass/fail result.

The README reports only numbers present in committed receipts.

## 18. README claim boundary

The first README should lead with the falsifiable question, not the brain metaphor.

Allowed if supported by receipts:

- "On this frozen corrupted-cue benchmark, iterative robust settling recovered X/Y native Qwen misses."
- "Temperature breathing alone did/did not improve retrieval."
- "The correct candidate was already assigned measurable support by the frozen model before settling."

Not allowed from v0 alone:

- "Brains use this algorithm."
- "This explains why LLMs are large."
- "Recurrence replaces parameter count."
- "Qwen thinks like a brain."

Those remain hypotheses or motivations.

## 19. First official decision table

| Outcome | Interpretation |
|---|---|
| Arm 4 <= native/fixed controls | Residue-settling idea fails in this formulation; preserve the kill |
| Arm 3 improves but arm 4 does not | Temperature schedule, not residue, is doing the work |
| Arm 4 improves corrupt but harms clean badly | Mechanism is an aggressive denoiser, not robust retrieval |
| Arm 4 improves corrupt and preserves clean | Proceed to unchanged-prompt / internal-attention gates |
| Gate 1 works, Gate 3 fails | External robust inference works; internal-attention story is unsupported |
| Gate 3 changes preference without new tokens and beats matched controls | Strong evidence that inference dynamics can recover frozen-model information missed by the native decision rule |

## 20. Success criterion for v0

A useful v0 is not necessarily a positive result. It is successful if another person can clone the repo and reproduce a controlled answer to:

> Given fixed Qwen weights and deliberately contradictory cues, does iterative broad/sharp retrieval plus residue-dependent trust recover correct candidates more robustly than native or matched iterative controls?

The repo ships the answer, including a negative one, with receipts.