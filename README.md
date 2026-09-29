# Breathing-Qwen

**Can a frozen language model recover an answer it weakly contains when one cue is confidently wrong?**

Breathing-Qwen is a controlled retrieval experiment around frozen **Qwen3-8B**. It started from a rhythmic-settling idea, but the experiments narrowed the useful mechanism: contradictory evidence leaves a residue that is better used to **choose the next computation** than to directly generate the answer from decomposed clues.

The current surviving recipe is:

```text
decompose to diagnose -> identify one suspicious cue -> re-query intact joint Qwen
```

The repository keeps negative results and frozen gates. A failed mechanism is not retuned away after seeing its result.

## Current status

| Gate | Status | Main result |
|---|---|---|
| Gate 0 | **FAIL** | residue improved corrupt accuracy by +5.47 pp, below the frozen +10 pp threshold |
| Gate 1-Q4 | **FAIL** | residue helped decomposed evidence but produced 3 recoveries and 6 new native errors |
| Gate 1B | **PASS — development set** | residue-guided re-query reached 93.75% corrupt accuracy, 6 recoveries, 0 new errors |
| Gate 1C | **NOT RUN — held-out** | frozen 128-item independent test of the unchanged Gate-1B mechanism |
| Gate 3 | boundary only | true internal attention-temperature intervention remains untested |

`Q4` / `NF4` means bitsandbytes 4-bit loading. Quantization and exact model revision are recorded in receipts rather than silently treated as full precision.

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

The frozen 32-item development benchmark SHA-256 is:

```text
38322de99fc14af71966dad1ecdcaa5f972300b2c3814455912440408f1a0f96
```

The complete NF4 run used:

```text
Qwen/Qwen3-8B
revision b968826d9c46dd6066d109eabc6255188de91218
```

| Arm | Clean | Corrupt |
|---|---:|---:|
| native all cues | **90.6%** | **75.0%** |
| per-cue one-shot | 71.9% | 50.0% |
| fixed | 71.9% | 50.0% |
| breathing-only | 71.9% | 50.0% |
| breathing + residue | 75.0% | 65.6% |

Residue beat the matched decomposed controls by **+15.6 pp** on corrupt items, but relative to native joint Qwen it produced only **3 recoveries and 6 new errors**. The frozen gate therefore failed.

The useful diagnosis was representational:

```text
F(A + B + C + D) != F(A) + F(B) + F(C) + F(D)
```

Splitting the query into separately scored clues loses information that joint conditioning uses. Yet the final residue weights located the deliberately corrupted cue on **30/32** items. That post-run observation motivated Gate 1B.

External beta “breathing” itself did no useful selection work here: fixed, per-cue one-shot and breathing all had the same 50% corrupt accuracy. Positive beta changes entropy and margin but not ordering of a fixed candidate score vector.

## Gate 1B: PASS on the development set

Gate 1B changes the role of residue:

> Use decomposition for **diagnosis**, then use intact joint Qwen for **reconstruction**.

For each query it compares:

1. `native_all_cues` — all clues jointly;
2. `random_remove` — remove one deterministic pseudo-random clue, then re-query jointly;
3. `residue_guided` — remove the lowest-trust residue clue, then re-query jointly;
4. `oracle_remove` — remove the known injected corrupt slot;
5. `exhaustive_leave_one_out` — average candidate probability vectors from all four one-clue-removed joint prompts.

The frozen controller constants stayed unchanged:

```text
beta  = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
delta = 0.20
trust floor = 0.05
```

Results on the same 32-item development benchmark:

| Arm | Clean | Corrupt | Mean corrupt rank |
|---|---:|---:|---:|
| native all cues | 90.625% | 75.0% | 1.3125 |
| random remove | 87.5% | 59.375% | 1.6563 |
| residue guided | **90.625%** | **93.75%** | **1.0625** |
| oracle remove | 93.75% | **93.75%** | **1.0625** |
| exhaustive probability ensemble | 90.625% | 59.375% | 1.4375 |

The locator identified the injected corrupt slot on **30/32 = 93.75%** of corrupt queries. Residue-guided re-query gained **+18.75 pp over native**, **+34.375 pp over random removal**, had **0 clean loss**, recovered **6** native corrupt failures and introduced **0 new errors**. It passed every frozen Gate-1B criterion.

On this development set, residue-guided re-query also matched the oracle-removal corrupt accuracy and mean rank. That is interesting, but it is **not independent confirmation**: Gate 1B was designed after inspecting Gate 1 on these same 32 targets.

The two remaining residue-guided answer errors were also oracle-removal errors, so perfect knowledge of the corrupt slot did not rescue those items. The two locator mistakes did not change already-correct answers.

The frozen Gate-1B design is in `docs/experiments/gate1b_residue_guided_requery.md`.

## Gate 1C: held-out replication — NOT RUN

Gate 1C freezes the unchanged Gate-1B mechanism on a new benchmark authored without consulting Qwen outputs on those items.

Frozen benchmark:

```text
benchmarks/v1_holdout.jsonl
128 items
SHA-256 c0d45ec644f4755f8bfb879457a9b1ae1311c20b4697d0aead9dfb83510dafed
```

Composition:

- 16 domains × 8 items;
- 5 candidates and 4 true cues per item;
- corrupt condition replaces exactly one cue;
- corrupt position balanced at 32 items per slot;
- 32 orthographic/name corruptions and 96 semantic/attribute corruptions;
- no target answer reused from the 32-item v0 development benchmark.

Official model provenance is frozen to:

```text
Qwen/Qwen3-8B
revision b968826d9c46dd6066d109eabc6255188de91218
NF4 4-bit
```

The official gate evaluates only when **all 128** items complete on those exact benchmark bytes and that exact model provenance. Residue-guided must simultaneously:

- localize the injected corrupt cue on at least **75%** of items;
- beat native corrupt accuracy by at least **8/128 = 6.25 pp**;
- beat deterministic random removal by at least **8/128 = 6.25 pp**;
- lose at most **4/128 = 3.125 pp** clean accuracy versus native;
- have **recoveries minus new errors >= 8** relative to native corrupt answers.

After the first official Gate 1C run begins, the benchmark, hash, model provenance, controller constants, arm definitions and thresholds are permanently frozen. Any dataset issue discovered later is documented rather than edited out of this gate.

Full protocol: `docs/experiments/gate1c_holdout.md`.

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

Gate 1-Q4:

```bash
python scripts/run_qwen.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```

Gate 1B development-set re-query:

```bash
python scripts/run_requery.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```

Gate 1C held-out replication:

```bash
python scripts/run_holdout.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```

For a smoke test, add `--limit 2`. Partial Gate 1C runs can never evaluate the frozen decision.

On Windows, `.gitattributes` pins `benchmarks/*.jsonl` to LF so the frozen SHA does not change merely because Git checked out CRLF line endings.

## What is and is not supported

Supported by completed runs:

- contradictory cues can degrade native Qwen retrieval;
- per-cue decomposition itself loses substantial joint-conditioning information;
- residue reweighting improved decomposed evidence but failed Gate 1-Q4 as a final answer generator;
- on the 32-item development set, residue was a strong corruption locator and residue-guided native re-query passed its frozen gate, matching oracle-removal corrupt accuracy.

Not supported yet:

- that the Gate-1B result generalizes to the 128-item held-out Gate 1C benchmark;
- that the method handles multiple bad cues, open-ended answers or RAG;
- that true internal attention-temperature breathing changes retrieval;
- that this mechanism explains biological rhythm or human memory;
- that recurrence substitutes for model scale.

The stronger unresolved “breathing” experiment is still true **internal** attention-temperature intervention. The external beta controller changes only the sharpness of an already-fixed candidate score vector.

## Repository layout

```text
breathing_qwen/
  schedules.py          frozen beta schedule
  robust.py             robust cue weighting
  settling.py           external settling controller
  toy_memory.py         Gate-0 CPU falsifier
  benchmark.py          paired benchmark schema + frozen hashes
  qwen_score.py         teacher-forced Qwen candidate scoring + provenance
  receipts.py           Gate-1 runner and receipts
  requery.py            Gate-1B residue-guided native re-query engine
  holdout.py            Gate-1C held-out wrapper + frozen decision/provenance
  internal_attention.py Gate-3 intervention boundary
benchmarks/
  v0.jsonl              32-item development benchmark
  v1_holdout.jsonl      128-item held-out Gate-1C benchmark
docs/experiments/       frozen experiment specifications
receipts/               experiment receipts
scripts/                runnable experiment CLIs
tests/                  CPU/unit tests; Qwen integration is opt-in
```

The point of this repository is not to preserve a favored mechanism. It is to keep narrowing the claim until the surviving statement is exactly what the experiments support.
