# Residue-Guided Requery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen Gate 1B experiment that uses residue only to locate a suspicious clue, then re-runs native joint Qwen without that clue, with strong controls and explicit quantization provenance.

**Architecture:** Keep Gate 1 unchanged. Add a focused `requery.py` runner that reuses the existing scorer and `breathing_residue` detector, then performs joint-condition rescoring after clue removal. Add a separate CLI and receipt path so the post-hoc development gate cannot be confused with the original Gate 1.

**Tech Stack:** Python 3.11+, NumPy, pytest, Transformers/Qwen optional runtime.

**Spec:** `docs/experiments/gate1b_residue_guided_requery.md`

## Global Constraints

- Exact v0 benchmark hash: `38322de99fc14af71966dad1ecdcaa5f972300b2c3814455912440408f1a0f96`.
- Exact official item count: 32.
- Reuse Gate-1 schedule `[0.50, 1.75, 0.65, 2.00]`, phases `[open, close, open, close]`, delta `0.20`, trust floor `0.05`.
- No threshold tuning after the first Gate 1B run starts.
- Gate 1 code and decision rule remain unchanged.
- Quantization mode must be written into new receipts.

## Review Focus

- Tie handling when two clue weights are equal: deterministic lowest index.
- Random-removal control must be stable across process, platform, clean/corrupt view, and partial runs.
- Exhaustive leave-one-out must combine probability vectors rather than raw prompt log-probabilities.
- Partial or wrong-hash runs must never evaluate the frozen Gate 1B decision.
- NF4 provenance must be explicit while non-quantized scorers continue to work.

---

### Task 1: Freeze tests for Gate 1B behavior

**Files:**
- Create: `tests/test_requery.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `BenchmarkItem`, `settle`, and scorer `score_cue(text, candidates)`.
- Produces: failing expectations for `Gate1BConfig`, `run_gate1b`, `evaluate_gate1b`, deterministic random removal, five arms, and `scripts/run_requery.py --help`.

- [ ] **Step 1:** Add tests that pin all five arms, residue-selected clue removal, deterministic random index, exhaustive probability averaging, frozen decision thresholds, partial-run non-evaluation, and CLI flags including `--load-4bit`.
- [ ] **Step 2:** Run CI and verify RED because Gate 1B code/CLI do not yet exist.
- [ ] **Step 3:** Commit the failing tests.

### Task 2: Implement Gate 1B core

**Files:**
- Create: `breathing_qwen/requery.py`

**Interfaces:**
- Consumes: `BenchmarkItem`, `DEFAULT_SCHEDULE`, `settle`, scorer interface.
- Produces: `Gate1BConfig`, `deterministic_random_index(item_id, cue_count)`, `evaluate_gate1b(...)`, `run_gate1b(items, scorer, config)` returning `RunReceipt`.

- [ ] **Step 1:** Implement deterministic SHA-256 random removal using `gate1b-v0:<item.id>`.
- [ ] **Step 2:** Implement the five frozen arms and per-item diagnostics, caching score vectors by `(candidates, text)`.
- [ ] **Step 3:** Implement the frozen Gate 1B decision exactly as the spec states.
- [ ] **Step 4:** Run focused tests and verify GREEN.
- [ ] **Step 5:** Commit.

### Task 3: Add explicit inference provenance and CLI

**Files:**
- Modify: `breathing_qwen/qwen_score.py`
- Create: `scripts/run_requery.py`
- Modify: `scripts/run_qwen.py`
- Modify: `tests/test_qwen_score.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `QwenCandidateScorer.from_pretrained(..., load_4bit=...)`.
- Produces: scorer `inference_mode` property (`"nf4"` or `"native"`) and CLIs that pass it into receipts.

- [ ] **Step 1:** Add failing provenance tests.
- [ ] **Step 2:** Implement inference-mode metadata without altering scoring behavior.
- [ ] **Step 3:** Add `run_requery.py` with the same model/revision/memory/4-bit options as `run_qwen.py`.
- [ ] **Step 4:** Run focused tests and verify GREEN.
- [ ] **Step 5:** Commit.

### Task 4: Windows benchmark portability and public docs

**Files:**
- Create: `.gitattributes`
- Modify: `README.md`

**Interfaces:**
- Produces: LF checkout for `benchmarks/*.jsonl`; public run instructions and explicit `Gate 1-Q4: FAIL` / `Gate 1B: NOT RUN` status.

- [ ] **Step 1:** Add `.gitattributes` rule `benchmarks/*.jsonl text eol=lf`.
- [ ] **Step 2:** Update README without changing historical Gate-0/Gate-1 claims.
- [ ] **Step 3:** Run complete CPU suite and Gate-0 replay.
- [ ] **Step 4:** Commit.

### Task 5: Final verification

**Files:** no new files.

- [ ] **Step 1:** Verify full non-Qwen pytest suite passes.
- [ ] **Step 2:** Verify Gate-0 replay still succeeds as a runtime test and still reports its historical scientific FAIL.
- [ ] **Step 3:** Verify `scripts/run_requery.py --help` and `scripts/run_qwen.py --help` from a fresh checkout path.
- [ ] **Step 4:** Review branch diff for accidental Gate-1 changes; there must be none beyond provenance plumbing.
