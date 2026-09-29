# Breathing-Qwen

**Can a frozen language model recover an answer it weakly contains when one cue is confidently wrong?**

Breathing-Qwen is a controlled retrieval experiment around frozen **Qwen3-8B**. It started from a simple idea: contradictory evidence should leave a residue that can change the next computation instead of being blindly averaged away.

The repository keeps negative results. The first external “breathing” mechanism did **not** pass its gates. What survived is narrower and more interesting: residue reweighting appears much better at **locating suspicious evidence** than at generating the final answer from decomposed cues. Gate 1B tests whether that diagnostic can guide a fresh native Qwen re-query.

## Current status

| Gate | Status | Main result |
|---|---|---|
| Gate 0 | **FAIL** | residue improved corrupt accuracy by +5.47 pp, below the frozen +10 pp threshold |
| Gate 1-Q4 | **FAIL** | residue beat matched decomposed controls but damaged too many native-correct answers |
| Gate 1B | **NOT RUN** | frozen residue-guided native re-query experiment |
| Gate 3 | boundary only | true internal attention-temperature intervention still untested |

`Q4` means the run used bitsandbytes NF4 4-bit loading. Quantization is now recorded explicitly in receipts; it is not silently treated as a full-precision run.

## Gate 0: FAIL

The CPU associative-memory falsifier was frozen before running: 256 clean/corrupt pairs, seed `1729`. The residue arm had to beat the best control on corrupted queries by at least 10 percentage points while losing no more than 2 points on clean queries.

| Arm | Clean | Corrupt |
|---|---:|---:|
| one-shot | 80.08% | 41.80% |
| fixed iteration | 80.08% | 41.80% |
| breathing-only | 80.08% | 41.80% |
| breathing + residue | **80.86%** | **47.27%** |

The gain was **+5.47 pp**, so the frozen gate failed. The receipt remains committed at `receipts/gate0_seed1729.json`.

## Gate 1-Q4: FAIL

Gate 1 uses Qwen only as a deterministic semantic evidence scorer. Candidate answers are scored by full-string, length-normalized teacher-forced conditional log-likelihood with thinking disabled.

The frozen benchmark contains 32 clean/corrupt pairs and changes exactly one cue in each corrupt item. SHA-256:

```text
38322de99fc14af71966dad1ecdcaa5f972300b2c3814455912440408f1a0f96
```

A complete Qwen3-8B NF4 run on 29 Sep 2026 used model revision:

```text
b968826d9c46dd6066d109eabc6255188de91218
```

Results:

| Arm | Clean | Corrupt |
|---|---:|---:|
| native all cues | **90.6%** | **75.0%** |
| per-cue one-shot | 71.9% | 50.0% |
| fixed | 71.9% | 50.0% |
| breathing-only | 71.9% | 50.0% |
| breathing + residue | 75.0% | 65.6% |

The residue arm cleared the frozen corrupt-gain threshold versus the matched decomposed controls: **+15.6 pp**. It also did not lose clean accuracy versus those controls. But relative to native joint Qwen it produced only **3 recoveries and 6 new errors**, for a recovery advantage of **−3**; the frozen rule required at least +4. Therefore Gate 1 fails.

The important diagnosis is representational: splitting a joint query into separately scored clues loses information. In shorthand,

```text
F(A + B + C + D) != F(A) + F(B) + F(C) + F(D)
```

Residue partially repairs the decomposed representation, but native joint conditioning remains much stronger.

A post-run diagnostic found that the lowest final residue-trust clue matched the deliberately corrupted slot on **30/32** items. On the paired clean versions, that same slot was the minimum only **7/32** times. This observation motivated Gate 1B; because it was discovered on this benchmark, it is not an independent result and is not part of the Gate 1 pass rule.

External beta “breathing” itself did no useful selection work in Gate 1: fixed, per-cue one-shot, and breathing all produced the same 50% corrupt accuracy. Positive beta changes entropy and margin but not candidate ordering when the evidence vector is fixed.

## Gate 1B: residue-guided native re-query — NOT RUN

Gate 1B freezes a different hypothesis:

> Use decomposition for **diagnosis**, then use intact joint Qwen for **reconstruction**.

For each clean/corrupt query it compares five arms:

1. `native_all_cues` — score all clues jointly.
2. `random_remove` — remove one deterministic pseudo-random clue, then score the rest jointly.
3. `residue_guided` — run the frozen Gate-1 residue detector, remove the lowest-trust clue, then score the remaining clues jointly.
4. `oracle_remove` — remove the known injected slot; an upper-bound control, not a deployable method.
5. `exhaustive_leave_one_out` — run every one-clue-removed joint prompt and average their candidate probability vectors.

The Gate-1 schedule and robust constants are unchanged:

```text
beta  = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
delta = 0.20
trust floor = 0.05
```

The Gate 1B pass rule was frozen before its first run. On the exact 32-item benchmark, residue-guided re-query must simultaneously:

- beat native corrupt accuracy by at least **2/32 = 6.25 pp**;
- beat deterministic random removal by at least **2/32 = 6.25 pp**;
- lose at most **1/32 = 3.125 pp** clean accuracy versus native;
- produce at least **2 more recoveries than new errors** relative to native corrupt answers.

Because Gate 1B was designed after inspecting Gate 1 on these same prompts, a pass is development-set evidence only. It would justify a new held-out benchmark; it would not count as independent confirmation.

The frozen design is in `docs/experiments/gate1b_residue_guided_requery.md`.

## Run it

Install:

```bash
python -m pip install -e ".[test,qwen]"
python -m pip install -U bitsandbytes
```

CPU tests:

```bash
python -m pytest -m "not qwen" -v
```

Gate 0:

```bash
python scripts/run_toy.py --json
```

Gate 1 on 4-bit Qwen3-8B:

```bash
python scripts/run_qwen.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```

Gate 1B on the same frozen model/benchmark:

```bash
python scripts/run_requery.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```

For a smoke test, add `--limit 2`. Partial runs can never evaluate the frozen Gate 1B decision.

On Windows, `.gitattributes` pins `benchmarks/*.jsonl` to LF so the frozen SHA does not change merely because Git checked out CRLF line endings.

## What is and is not being tested

Supported by the current runs:

- contradictory cues can degrade native Qwen retrieval;
- per-cue decomposition itself loses substantial joint-conditioning information;
- residue reweighting improves the decomposed representation but failed the frozen Gate 1 decision;
- the residue signal is promising as a corruption locator on the development benchmark.

Not supported yet:

- that residue-guided native re-query improves Qwen retrieval;
- that internal attention-temperature breathing changes what Qwen can retrieve;
- that this mechanism explains biological rhythm or human memory;
- that recurrence substitutes for model scale.

The stronger unresolved experiment is still true **internal** attention-temperature intervention. The external beta controller in Gate 1 changes only the sharpness of an already-fixed candidate score vector.

## Repository layout

```text
breathing_qwen/
  schedules.py          frozen beta schedule
  robust.py             robust cue weighting
  settling.py           external settling controller
  toy_memory.py         Gate-0 CPU falsifier
  benchmark.py          paired benchmark schema
  qwen_score.py         teacher-forced Qwen candidate scoring + inference provenance
  receipts.py           Gate-1 runner and receipts
  requery.py            Gate-1B residue-guided native re-query
  internal_attention.py Gate-3 intervention boundary
benchmarks/v0.jsonl     frozen 32-pair benchmark
docs/experiments/       frozen experiment specifications
receipts/               committed official receipts
scripts/                runnable experiment CLIs
tests/                  CPU/unit tests; Qwen integration is opt-in
```

The point of this repository is not to preserve a favored mechanism. It is to keep changing the mechanism until the surviving claim is exactly what the experiments support.
