# Breathing-Qwen v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible experiment that tests whether broad/sharp retrieval plus residue-dependent cue trust can recover correct candidates from deliberately contradictory cues better than matched controls, first in a CPU reference model and then with frozen Qwen3-8B scoring.

**Architecture:** Keep Qwen frozen and separate semantic evidence extraction from the settling controller. The controller operates only on cue-by-candidate score matrices, so Gate 0, Gate 1, and later internal-attention experiments share one deterministic inference core. Every official result is emitted as a receipt and README claims must come from committed receipts.

**Tech Stack:** Python 3.11+, NumPy, pytest; optional PyTorch + Transformers + Accelerate for Qwen3-8B integration.

**Spec:** `docs/superpowers/specs/2026-09-29-breathing-qwen-v0-design.md`

## Global Constraints

- Primary model is `Qwen/Qwen3-8B`; official Qwen receipts pin an exact revision.
- Model weights remain frozen; no training or fine-tuning in v0.
- Generation is deterministic; generation temperature is not the intervention.
- Default breathing schedule is `beta=[0.50, 1.75, 0.65, 2.00]` with phases `open, close, open, close`.
- Controller never receives the correct answer or `corrupt_index` during inference.
- CI never downloads Qwen3-8B.
- Official benchmark/configuration is frozen before the first official Qwen run.
- Gate 1 claims external robust settling only; Gate 3 internal-attention claims remain separate.
- Negative results and failed gates are preserved rather than rewritten.

## Review Focus

- A cue score row containing NaN/Inf must fail the item/run explicitly rather than be silently normalized.
- Candidate sets containing multi-token answers must use full-string, length-normalized conditional log-likelihood.
- A cue with extreme disagreement must lose influence without receiving exactly zero trust.
- Reordering cues must not change results in the external-settling controller except for explicitly order-sensitive controls.
- A missing local model / insufficient accelerator memory must produce a clear integration error without breaking Gate 0 or unit tests.

---

### Task 1: Package skeleton, schedules, and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `breathing_qwen/__init__.py`
- Create: `breathing_qwen/config.py`
- Create: `breathing_qwen/schedules.py`
- Create: `tests/test_schedules.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: `BreathingSchedule(betas: tuple[float, ...], phases: tuple[str, ...])`
- Produces: `DEFAULT_SCHEDULE` with the exact four values from the spec.
- Produces: immutable experiment configuration dataclasses used by later tasks.

- [ ] **Step 1: Write failing schedule/config tests**

Assert the default schedule equals `(0.50, 1.75, 0.65, 2.00)`, has four phases `("open", "close", "open", "close")`, rejects non-positive beta values, and rejects length mismatches.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_schedules.py -v`
Expected: FAIL because the package/API does not yet exist.

- [ ] **Step 3: Implement package metadata, immutable config types, and schedule validation**

Keep runtime dependencies minimal: NumPy in the base install; Qwen dependencies in an optional `qwen` extra.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/test_schedules.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add experiment configuration and breathing schedule`

---

### Task 2: Residue, robust cue trust, and deterministic settling controller

**Files:**
- Create: `breathing_qwen/robust.py`
- Create: `breathing_qwen/settling.py`
- Create: `tests/test_robust.py`
- Create: `tests/test_settling.py`

**Interfaces:**
- Consumes: `BreathingSchedule` from Task 1.
- Produces: `huber_weights(residuals: np.ndarray, *, delta: float, floor: float) -> np.ndarray`
- Produces: `settle(evidence: np.ndarray, schedule: BreathingSchedule, *, mode: str, delta: float, weight_floor: float) -> SettlingTrace`
- `evidence` shape is `(n_cues, n_candidates)`; the controller has no labels.
- `SettlingTrace` exposes per-cycle aggregate scores, probabilities, entropy, residuals, cue weights, winner index, and final margin.

- [ ] **Step 1: Write failing robust-weight tests**

Pin these properties: zero/small residuals retain maximal weight; larger residuals never gain weight over smaller residuals; all weights stay within `[floor, 1]`; NaN/Inf raises `ValueError`; an extreme outlier never receives zero trust.

- [ ] **Step 2: Write failing settling tests**

Use a tiny fixed evidence matrix to assert: one-shot is one aggregation; fixed-temperature iteration does not mutate trust; breathing-only follows the schedule with frozen trust; breathing+residue updates trust; cue-order permutation leaves the final external-settling result invariant after inverse permutation of trace rows; no label fields are accepted by `settle`.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `pytest tests/test_robust.py tests/test_settling.py -v`
Expected: FAIL because implementations do not exist.

- [ ] **Step 4: Implement normalized Huber/IRLS weighting and the four controller arms**

Modes are exactly: `one_shot`, `fixed`, `breathing`, `breathing_residue`. Softmax must be numerically stable. Residue is computed from disagreement between each cue score vector and the current aggregate after per-row centering/scale normalization so raw logit magnitude alone cannot define trust.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `pytest tests/test_robust.py tests/test_settling.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add residue-weighted settling controller`

---

### Task 3: CPU Gate 0 associative-memory falsifier

**Files:**
- Create: `breathing_qwen/toy_memory.py`
- Create: `scripts/run_toy.py`
- Create: `tests/test_toy_memory.py`

**Interfaces:**
- Consumes: `settle(...)` from Task 2.
- Produces: `make_toy_benchmark(seed: int, *, n_items: int, n_memories: int, dim: int) -> ToyBenchmark`
- Produces: `run_toy_gate(config: ToyGateConfig) -> dict`
- CLI emits JSON and exits nonzero only for invalid configuration/runtime failure, not because a scientific gate is negative.

- [ ] **Step 1: Write the failing seeded benchmark tests**

Assert exact reproducibility for a frozen seed, distinct clean/corrupt views of the same item, exactly one injected misleading feature group per corrupt item, and identical stored memories across all arms.

- [ ] **Step 2: Add failing Gate-0 metric tests**

Assert output contains clean accuracy, corrupt accuracy, corruption penalty, native-miss recovery rate, clean regression rate, margins, and per-arm pass/fail metadata. Add a test that an intentionally unhelpful parameterization can report a negative scientific result without raising an exception.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/test_toy_memory.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement the synthetic memory, corruption process, arm runner, and CLI**

Freeze the first official Gate-0 seed/config in code. The benchmark is deliberately simple: correlated feature groups define stored objects, one group is corrupted toward a hard distractor, and no corruption label is passed to the controller.

- [ ] **Step 5: Run Gate 0 and record the first receipt candidate**

Run: `python scripts/run_toy.py --json`
Expected: valid JSON for all four arms. Do not tune parameters after observing the official seeded outcome; any exploratory changes use a separate development seed/config.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_toy_memory.py tests/test_robust.py tests/test_settling.py -v`
Expected: PASS.
Commit message: `feat: add cpu gate zero falsifier`

---

### Task 4: Frozen benchmark schema and v0 corrupted-cue set

**Files:**
- Create: `breathing_qwen/benchmark.py`
- Create: `benchmarks/v0.jsonl`
- Create: `tests/test_benchmark_schema.py`

**Interfaces:**
- Produces: `BenchmarkItem` with `id`, `answer`, `clean_cues`, `corrupt_cues`, `corrupt_index`, and `candidates`.
- Produces: `load_benchmark(path: Path) -> list[BenchmarkItem]`.
- Inference-facing helpers expose cues/candidates only; evaluation helpers may expose labels after prediction.

- [ ] **Step 1: Write failing schema/blinding tests**

Assert unique IDs, answer occurs exactly once in candidates, clean/corrupt cue arrays have equal lengths, exactly one cue differs per pair, `corrupt_index` names that position, candidate lists are identical across the pair, and the inference view omits both answer and corrupt index.

- [ ] **Step 2: Run schema tests and verify failure**

Run: `pytest tests/test_benchmark_schema.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement schema parser and commit the 32 paired v0 items**

Cover anatomy/science, common objects, people/places, technical vocabulary, false-initial cues, category conflicts, and plausible false associations. Avoid items whose false cue is trivially detectable from syntax alone.

- [ ] **Step 4: Run schema tests and freeze benchmark hash**

Run: `pytest tests/test_benchmark_schema.py -v`
Expected: PASS. Record SHA-256 of the exact JSONL bytes for receipts.

- [ ] **Step 5: Commit**

Commit message: `test: freeze v0 corrupted cue benchmark`

---

### Task 5: Qwen3-8B deterministic candidate scorer

**Files:**
- Create: `breathing_qwen/qwen_score.py`
- Create: `tests/test_qwen_score.py`
- Create: `tests/test_qwen_smoke.py`

**Interfaces:**
- Produces: protocol `EvidenceScorer.score_cue(cue: str, candidates: Sequence[str]) -> np.ndarray`.
- Produces: `QwenCandidateScorer` implementing full-answer, length-normalized teacher-forced conditional log-likelihood.
- Qwen import/model load occurs lazily so base tests run without Transformers or model weights.

- [ ] **Step 1: Write failing scorer-contract tests using a fake tokenizer/model**

Assert deterministic scores, full multi-token candidate accounting, length normalization, stable candidate ordering, no free-generation call, explicit NaN/Inf rejection, and lazy optional dependency behavior.

- [ ] **Step 2: Write opt-in integration smoke test**

Mark with `pytest.mark.qwen`; skip cleanly unless a local model path/revision is explicitly supplied. Smoke test scores a two-candidate cue twice and asserts equal finite outputs.

- [ ] **Step 3: Run base tests and verify failure**

Run: `pytest tests/test_qwen_score.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement scorer and constrained-memory loader**

Support `device_map="auto"`, optional `max_memory`, exact revision recording, and explicit thinking-disabled prompt formatting where supported. Never download an 8B model during ordinary test collection.

- [ ] **Step 5: Run base scorer tests and commit**

Run: `pytest tests/test_qwen_score.py -v`
Expected: PASS.
Commit message: `feat: add frozen qwen candidate scorer`

---

### Task 6: Gate 1 runner, metrics, receipts, and summary CLI

**Files:**
- Create: `breathing_qwen/receipts.py`
- Create: `scripts/run_qwen.py`
- Create: `scripts/summarize_receipts.py`
- Create: `tests/test_receipts.py`
- Create: `tests/test_gate1_runner.py`
- Create: `receipts/.gitkeep`

**Interfaces:**
- Consumes: benchmark loader, `EvidenceScorer`, and `settle`.
- Produces: `run_gate1(items, scorer, config) -> RunReceipt`.
- Produces: JSON-serializable `RunReceipt` containing environment, model revision, benchmark hash, schedule, robust constants, arm metrics, per-item predictions, and full settling traces.
- Produces: summary CLI that reads receipts only; it never recomputes model scores.

- [ ] **Step 1: Write failing runner/metric tests with a deterministic fake scorer**

Pin corruption penalty, native-miss recovery rate, clean regression rate, mean rank/margin, cue trust trajectories, known-corrupt-cue trust diagnostic computed only after inference, and cue-order control behavior.

- [ ] **Step 2: Write failing receipt tests**

Assert JSON round-trip, required provenance fields, benchmark SHA, exact schedule/constants, no hidden chain-of-thought text field, and preservation of partial item results when an integration error occurs.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/test_gate1_runner.py tests/test_receipts.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement Gate 1 runner and receipt serializer**

Cache per-cue Qwen score vectors once and reuse the exact evidence across all controller arms. Include shuffled/reordered cue controls without rescoring text when possible.

- [ ] **Step 5: Implement CLIs and dry-run with fake/synthetic evidence**

Run: `python scripts/run_qwen.py --help` and `python scripts/summarize_receipts.py --help`.
Expected: usable CLI help without loading Qwen.

- [ ] **Step 6: Run all non-Qwen tests and commit**

Run: `pytest -m "not qwen" -v`
Expected: PASS.
Commit message: `feat: add gate one runner and scientific receipts`

---

### Task 7: README, official run protocol, and optional Gate 2/3 adapter boundary

**Files:**
- Create: `README.md`
- Create: `breathing_qwen/internal_attention.py`
- Create: `tests/test_internal_attention.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `InternalAttentionAdapter` compatibility boundary that is disabled by default and fails clearly when the installed Transformers/Qwen implementation cannot expose pre-softmax attention scores.
- README reports no positive Gate-1/2/3 result until a committed receipt exists.

- [ ] **Step 1: Write failing adapter-boundary tests**

Assert unsupported model/version raises a versioned compatibility error; beta/span-bias intervention is a no-op outside explicitly selected layers/heads; default import does not patch global Transformers state.

- [ ] **Step 2: Implement only the safe adapter boundary, not an unverified model-specific hook**

Expose the interface needed for a later selected-head implementation while keeping Gate 1 independent. If a stable hook is available during execution, add it behind explicit opt-in and corresponding tests; otherwise preserve Gate 3 as unsupported rather than faking it.

- [ ] **Step 3: Add CI for Python unit/Gate-0 tests only**

Run NumPy/base tests on CPU; explicitly exclude `qwen` marker and any 8B download.

- [ ] **Step 4: Write README around the falsifiable question and run protocol**

Include: hypothesis, four/five controls, exact distinction between generation and retrieval temperature, Gate ladder, commands for Gate 0 and Qwen, receipt policy, and a results section that starts as `Not run yet` for Qwen gates.

- [ ] **Step 5: Run repository verification**

Run: `pytest -m "not qwen" -v`
Run: `python scripts/run_toy.py --json`
Expected: tests pass and Gate-0 JSON is produced. Scientific Gate 0 may be positive or negative; README language must match the receipt rather than expectation.

- [ ] **Step 6: Commit**

Commit message: `docs: finish reproducible breathing qwen v0 harness`

---

## First Qwen run after implementation

This is an execution protocol, not a coding task:

1. Pin the exact local `Qwen/Qwen3-8B` revision before scoring.
2. Confirm `benchmarks/v0.jsonl` and controller constants are unchanged from their committed hashes.
3. Run a two-item smoke test first for memory/runtime validation only; do not inspect/tune scientific outcomes.
4. Run the full 32-pair Gate 1 once and save the timestamped receipt.
5. Commit the receipt unchanged.
6. Update README results only from that receipt.
7. Only if Gate 1 supports proceeding, attempt Gate 2 and then the selected-head internal Gate 3.

## Completion criteria

The implementation is complete when:

- `pytest -m "not qwen" -v` passes without model downloads;
- Gate 0 produces a deterministic receipt;
- the 32-pair benchmark is schema-valid and hash-frozen;
- Qwen scoring is deterministic and teacher-forced;
- every arm uses identical cached cue evidence;
- official runs emit provenance-complete receipts;
- README claims do not exceed the strongest completed gate;
- a user with local Qwen3-8B can run Gate 1 from one documented command.
