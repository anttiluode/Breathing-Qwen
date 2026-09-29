# Gate 1C — held-out residue-guided re-query

**Status at freeze:** NOT RUN.

Gate 1C is the independent follow-up to the 32-item Gate 1B development-set result. Gate 1B was designed after inspecting Gate 1 on `benchmarks/v0.jsonl`, so its 93.75% locator accuracy and 93.75% residue-guided corrupt accuracy are development evidence only. Gate 1C asks whether the same mechanism survives new targets without retuning.

## Frozen hypothesis

Use the decomposed per-cue representation only to locate the least-compatible cue, then return to intact joint Qwen inference after removing that one cue.

```text
per-cue evidence -> residue locator -> choose one cue -> native joint re-query
```

No controller constant or re-query rule is changed from Gate 1B.

## Frozen model provenance

The official gate is evaluated only for:

```text
model: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218
inference mode: bitsandbytes NF4 4-bit
thinking: disabled by the candidate-scoring prompt path
```

A run with another model, revision, or inference mode may produce a receipt but cannot evaluate the official Gate 1C decision.

## Frozen benchmark

File:

```text
benchmarks/v1_holdout.jsonl
```

SHA-256:

```text
c0d45ec644f4755f8bfb879457a9b1ae1311c20b4697d0aead9dfb83510dafed
```

Composition:

- 128 items;
- 16 domains with 8 items each: neuroscience, cell biology, chemistry, physics, astronomy, geography, history, literature, music, art, computing, mathematics, instruments, animals, plants/ecology, and everyday objects;
- 5 candidate answers per item;
- 4 clean cues per item;
- corrupt condition replaces exactly one of those four cues;
- corrupt slot is exactly balanced: 32 items each at positions 0, 1, 2, and 3;
- 32 corruptions are orthographic/name cues; the remaining 96 use semantic attributes such as function, category, location, composition, identity, or mechanism;
- no target answer is reused from the 32-item v0 development benchmark.

The benchmark was authored and frozen without consulting Qwen outputs on these items. Structural tests pin the exact bytes, size, domain balance, slot balance, and target non-overlap.

## Frozen mechanism and controls

Gate 1C calls the same Gate-1B scoring/re-query engine and compares the same five arms:

1. `native_all_cues` — all four cues scored jointly.
2. `random_remove` — remove one deterministic pseudo-random cue, then score the remaining three jointly.
3. `residue_guided` — use the frozen per-cue residue controller to select the lowest-trust cue, remove it, then score the remaining three jointly.
4. `oracle_remove` — remove the known injected corrupt slot; upper-bound control only.
5. `exhaustive_leave_one_out` — score all four one-cue-removed joint prompts and average their candidate probability vectors.

Frozen residue constants:

```text
beta  = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
delta = 0.20
trust floor = 0.05
```

The residue detector does not generate the answer. It selects the next operator/input; the final answer comes from a fresh native joint Qwen score over the surviving cues.

## Frozen Gate 1C decision

All 128 items must complete on the exact frozen benchmark and exact frozen model provenance. `residue_guided` must simultaneously satisfy:

- corruption localization accuracy **>= 75%** (>= 96/128);
- corrupt accuracy gain over `native_all_cues` **>= 8/128 = 6.25 percentage points**;
- corrupt accuracy gain over `random_remove` **>= 8/128 = 6.25 percentage points**;
- clean accuracy loss versus native **<= 4/128 = 3.125 percentage points**;
- recoveries minus new errors relative to native corrupt answers **>= 8**.

The proportional accuracy thresholds match Gate 1B; the recovery count is scaled from 2/32 to 8/128. Locator accuracy is an additional explicit requirement because corruption localization is now the central mechanism claim.

## Interpretation boundary

A pass would support this narrow statement:

> On a new 128-item benchmark with one injected contradictory cue, the frozen residue signal can locate corruption well enough to guide a native joint re-query that improves retrieval without unacceptable clean degradation.

A pass would **not** establish that:

- the mechanism handles multiple simultaneous false cues;
- it works without candidate lists;
- it generalizes to open-ended generation or RAG;
- attention-temperature breathing works;
- the mechanism is biologically implemented.

A failure is equally informative and remains the result.

## No-edit rule

After the first official Gate 1C model run begins, `benchmarks/v1_holdout.jsonl`, its hash, model provenance, controller constants, arm definitions, and decision thresholds are frozen permanently. Any factual ambiguity or dataset error discovered afterward is documented in the receipt/README and corrected only in a new benchmark version; it is not edited away from Gate 1C.

## Run

Smoke test (cannot evaluate the gate):

```bash
python scripts/run_holdout.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit \
  --limit 2
```

Official run:

```bash
python scripts/run_holdout.py \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --candidate-batch-size 1 \
  --load-4bit
```
