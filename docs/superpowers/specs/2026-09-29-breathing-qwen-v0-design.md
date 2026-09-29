# Breathing-Qwen v0 — Design

Date: 2026-09-29  
Status: draft for written-spec review  
Repository: `anttiluode/Breathing-Qwen`

## 1. Purpose

Breathing-Qwen tests one narrow hypothesis:

> A frozen language model can sometimes contain enough information to recover an answer that its ordinary one-pass retrieval misses, and an iterative settling process that alternates broad and sharp retrieval while reducing trust in inconsistent cues can recover that answer without adding new external evidence.

The project is not a claim about brain rhythms, consciousness, or a universal replacement for model scale. The neuroscience analogy motivates the mechanism; the repo tests the mechanism.

The initial target is `Qwen/Qwen3-8B` because the earlier local Qwen work already established a practical constrained-memory setup around this model family.

## 2. The distinction that must stay clean

1. **Generation temperature** controls randomness while sampling output tokens. Breathing-Qwen does not use this as the intervention; official evaluation is deterministic.
2. **Retrieval temperature** (`beta`, inverse temperature) changes how sharply candidates or attention compete before an answer is emitted.
3. **Residue-dependent trust** changes which cues or source spans are allowed to influence later retrieval cycles.

The comparison is therefore not "hot Qwen versus cold Qwen." It is fixed weights and fixed evidence under different inference dynamics.

## 3. Hypotheses

### H0 — breathing alone

Oscillating retrieval sharpness without changing trust does not reliably improve corrupted-cue retrieval over a fixed-temperature iterative baseline.

### H1 — residue changes the next computation

When one cue is inconsistent with the rest of a partially specified object, an iterative robust estimator can reduce that cue's authority and recover the correct answer more often than native one-pass retrieval or breathing temperature alone.

### H2 — frozen-model recoverability

For at least some failures, frozen Qwen3-8B assigns recoverable support to the correct candidate before decoding even when the ordinary all-cues decision is wrong. Iterative settling can expose that support without training the model and without generating reasoning tokens between cycles.

### H3 — internal intervention, later gate

If retrieval temperature and source-span trust are applied inside selected Qwen attention heads, candidate preference may change across repeated passes over an unchanged prompt. This is a stronger claim than external settling and must be reported separately.

## 4. Non-goals

v0 will not:

- train or fine-tune Qwen;
- claim theta rhythm literally implements softmax temperature;
- claim a smaller recurrent model can replace an 8B model;
- use chain-of-thought tokens as the settling loop;
- benchmark general reasoning ability;
- hide negative results;
- require a large GPU for CI.

## 5. Experimental ladder

The project is staged so a failure at a later gate does not invalidate an earlier measurement.

### Gate 0 — reference dynamical memory (CPU, no LLM)

Implement a small NumPy associative-retrieval system with stored vectors and corrupted query features.

Four arms use identical memories and observations:

1. one-shot retrieval;
2. repeated fixed-temperature retrieval;
3. breathing temperature only;
4. breathing temperature plus residue-dependent trust.

Minimal form:

```text
p_t = softmax(beta_t * score(S, W_t, cues))
residue_t[j] = mismatch(cue_j, retrieved_state_t)
W_{t+1}[j] = robust_weight(residue_t[j])
```

`W` is bounded away from zero so a cue can lose authority without being silently deleted.

**Frozen Gate-0 receipt:** 256 clean/corrupt paired synthetic trials, seed `1729`. Arm 4 passes only if corrupted accuracy is at least **10 percentage points** above the best of arms 1–3 and clean accuracy is no more than **2 percentage points** below the best clean control. These constants are part of v0 and are not tuned after the run.

### Gate 1 — Qwen external settling (primary v0)

Use frozen Qwen3-8B as a semantic scoring engine. No internal Qwen code is changed for this gate.

Each benchmark item contains:

- an answer;
- four individually addressable cues;
- a clean condition;
- a paired corrupt condition in which exactly one cue is replaced by a plausible false cue;
- five candidate answers containing the target and four hard distractors.

For each cue, Qwen produces a deterministic score vector over the same five candidates. These per-cue vectors are evidence objects. The controller aggregates them iteratively:

```text
z_j       = frozen Qwen candidate-score vector from cue j
mu_t      = weighted aggregate(z_1 ... z_n; W_t)
p_t       = softmax(beta_t * mu_t)
residue_j = distance(z_j, mu_t)
W_{t+1}   = robust weights(residue)
```

No corruption label or correct answer enters the update. Labels are used only after inference for evaluation.

Gate 1 is explicitly a **model-external robust inference** result. If it works, it does not prove Qwen attention itself performs the settling dynamics.

**Frozen Gate-1 benchmark:** 32 paired clean/corrupt items. Arm 4 passes only if all three conditions hold:

- corrupt accuracy is at least **10 percentage points** above the best matched control (arms 2–3);
- clean accuracy is no more than **3 percentage points** below the best matched control;
- on native-Qwen misses, arm 4 produces at least **4 more recoveries than new errors**.

The 32 official pairs are frozen before the first official Qwen receipt. Exploratory tuning, if needed, uses a separate eight-item development set that never contributes to the official metrics.

### Gate 2 — unchanged-prompt multi-pass Qwen

Run repeated deterministic forward passes over one unchanged prompt. No tokens are generated between passes. Span-level cue weights are carried by the controller from one pass to the next.

The first implementation may express span trust through a controlled score/bias mechanism if a stable internal attention hook is unavailable. Receipts must name the exact intervention rather than calling it internal attention.

The required observation is a candidate-preference trajectory such as:

```text
cycle 0: distractor > target
cycle 1: distractor ~= target
cycle 2: target > distractor
```

with unchanged model weights and unchanged external evidence.

### Gate 3 — internal attention breathing (stretch gate)

Force an attention implementation that exposes pre-softmax attention scores. For a small predeclared set of layers/heads, replace

```text
softmax(scores)
```

with

```text
softmax(beta_t * scores + span_bias(W_t))
```

All unselected heads remain native. Head/layer selection must come from a predeclared causal/span-sensitivity probe, never from final benchmark outcomes.

Gate 3 is optional for the first shipped v0. A clean Gate-1 result is more valuable than a fragile internal patch.

## 6. Why this ladder exists

It prevents three confounds:

- **sampling confound:** changing output temperature can look like exploration while merely adding randomness;
- **extra-information confound:** generated reasoning tokens change context, so the later answer no longer uses fixed evidence;
- **controller/model confound:** an external robust aggregator working does not prove Qwen internally implements the same dynamics.

Each gate increases the claim only after the simpler one is measured.

## 7. Benchmark

### 7.1 Item schema

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

The clean partner replaces the false cue with a true one. `corrupt_index` is unavailable to inference code.

### 7.2 Composition

The eight-item development set and 32-pair official set should span:

- anatomy/science terms;
- common objects;
- people/places with strong semantic associations;
- technical vocabulary;
- false category cues;
- false initial-letter cues;
- false but semantically plausible associations.

A single lexical corruption type may not exceed half the official set.

### 7.3 Frozen-before-run rule

Official items, schedule, seeds, pass thresholds, candidate sets, and arm definitions are committed before the first official Qwen run. Any later change creates a new benchmark version and receipt lineage instead of overwriting v0.

## 8. Metrics

Primary:

- candidate accuracy on clean items;
- candidate accuracy on corrupt items;
- corruption penalty (`clean - corrupt`);
- recovery rate on native-Qwen misses;
- new-error count on items native Qwen gets right;
- clean-regression rate;
- target rank and target-vs-best-distractor margin;
- number of model forward passes.

Mechanism:

- cue trust by cycle;
- whether the known corrupt cue loses trust (evaluation only);
- candidate margin by cycle;
- entropy across open/close phases;
- convergence / oscillation / no-decision rate.

Gate 3 additionally records attention mass on declared cue spans, selected head/layer IDs, and source K/V integrity hashes where applicable.

## 9. Required controls

Every official run includes:

1. native all-cues Qwen;
2. fixed per-cue aggregation, no iteration;
3. repeated fixed-temperature settling;
4. breathing temperature only;
5. breathing plus residue-dependent trust.

Required ablations for the first positive result:

- shuffled cue order;
- corrupt cue moved to a different position;
- trust update with `beta` fixed;
- beta breathing with trust frozen;
- random cue down-weighting with matched total weight loss.

This explicitly tests whether apparent recovery is merely a position/order correction.

## 10. Default breathing schedule

Frozen v0 schedule:

```text
beta  = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
```

The fixed-temperature control uses `beta = 1.0` for all four cycles. Schedule tuning on the official benchmark is prohibited.

## 11. Residue and robust weighting

The primary external-settling residue is disagreement between a cue-specific candidate-score vector and the current aggregate score vector.

v0 uses an explicit robust M-estimator/IRLS-style weighting rule with scale normalization. The implementation plan must choose the exact rule and constant before the first benchmark run; Huber weighting is the default unless implementation testing reveals a numerical reason it is unsuitable.

Requirements:

- no answer label enters the update;
- all weights remain inspectable;
- weights have a nonzero floor;
- the rule is deterministic;
- the exact constants appear in receipts;
- unit tests cover synthetic outliers and equal-evidence cases.

## 12. Qwen scoring

Primary model: `Qwen/Qwen3-8B`, pinned to an exact model revision in official receipts.

Candidate scoring uses teacher-forced conditional log-likelihood rather than free generation. Scores are length-normalized over answer tokens. This gives deterministic margins even when multiple arms would greedily emit the same text.

Thinking/reasoning output is disabled for the core experiment. No chain-of-thought text is needed or stored.

Runtime supports constrained local hardware using `device_map="auto"`, configurable `max_memory`, candidate micro-batching, and resumable per-item receipts. The default example configuration should accommodate the same approximately 6 GiB GPU / CPU-offload class of setup used in earlier local Qwen tests, without making that memory value a hard requirement.

## 13. Package layout

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
    dev_v0.jsonl
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
    test_qwen_smoke.py      # opt-in; no CI model download
  receipts/
    .gitkeep
  docs/superpowers/specs/
    2026-09-29-breathing-qwen-v0-design.md
```

## 14. Interfaces

### `BreathingSchedule`

Pure deterministic data: beta-by-cycle and phase labels.

### `EvidenceScorer`

Converts one cue into one vector of candidate scores. Gate 0 uses a synthetic scorer; Gate 1 uses frozen Qwen.

### `SettlingController`

Consumes cue-score vectors, schedule, and robust-weight rule. Produces the complete cycle trace: aggregate scores, probabilities, residue, and cue weights. It never receives answer labels or `corrupt_index`.

### `BenchmarkRunner`

Runs all arms over the same frozen items and emits machine-readable receipts.

### `InternalAttentionAdapter`

Optional Gate-3 boundary that owns all Transformers/Qwen-specific hooking so the core controller remains independent of model internals.

## 15. Error handling

- missing model or insufficient memory: fail clearly and preserve completed per-item receipts;
- NaN/Inf score: mark item invalid and fail the official aggregate rather than silently clip;
- candidate tokenization: score full answer strings, length-normalized, and record token counts;
- non-convergence: record explicitly; never force a winner;
- unsupported Gate-3 internals: Gate 0/1 remain runnable and Gate 3 exits with a versioned compatibility error.

## 16. Testing strategy

CI never downloads Qwen3-8B.

Unit tests cover:

- schedule determinism;
- robust-weight monotonicity and weight floor;
- residue invariants;
- controller blindness to labels;
- clean/corrupt schema pairing;
- deterministic seeded Gate-0 receipt;
- JSON receipt round-trip;
- CLI validation.

An opt-in integration test (`pytest -m qwen`) runs only when the model is locally available or explicit download permission is supplied.

Gate 0 is the CI scientific smoke test and must reproduce its frozen receipt within declared floating-point tolerances.

## 17. Receipts and provenance

Every run writes a timestamped JSON receipt containing:

- git commit SHA;
- Python, NumPy, PyTorch, and Transformers versions as applicable;
- model name and exact revision;
- device/memory configuration;
- benchmark version and content hash;
- seeds;
- arm configuration;
- breathing schedule;
- robust rule/constants;
- aggregate metrics;
- per-item predictions and cycle traces;
- gate thresholds and pass/fail result.

The README reports only numbers backed by committed receipts.

## 18. README claim boundary

Allowed if supported by receipts:

- "On this frozen corrupted-cue benchmark, iterative robust settling recovered X/Y native Qwen misses."
- "Temperature breathing alone did/did not improve retrieval."
- "The correct candidate had measurable frozen-model support before settling."

Not allowed from v0 alone:

- "Brains use this algorithm."
- "This explains why LLMs are large."
- "Recurrence replaces parameter count."
- "Qwen thinks like a brain."

Those remain motivations or future hypotheses.

## 19. Decision table

| Outcome | Interpretation |
|---|---|
| Arm 5 <= matched controls | Residue-settling idea fails in this formulation; preserve the kill |
| Arm 4 improves but arm 5 does not | Breathing schedule, not residue, is doing the work |
| Arm 5 improves corrupt but harms clean beyond threshold | Aggressive denoising, not robust retrieval |
| Arm 5 passes Gate 1 | Proceed to unchanged-prompt / internal-attention gates |
| Gate 1 works, Gate 3 fails | External robust inference works; internal-attention story is unsupported |
| Gate 3 changes preference without new tokens and beats matched controls | Evidence that altered inference dynamics can recover frozen-model support missed by the native decision rule |

## 20. Success criterion

A useful v0 is not necessarily positive. It succeeds as a research artifact if another person can clone the repo and reproduce a controlled answer to:

> Given fixed Qwen weights and deliberately contradictory cues, does iterative broad/sharp retrieval plus residue-dependent trust recover correct candidates more robustly than native or matched iterative controls?

The repo ships the answer, including a negative one, with receipts.