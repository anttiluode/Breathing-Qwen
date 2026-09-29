# Breathing-Qwen

**Can a frozen language model recover an answer it weakly contains when one cue is confidently wrong?**

Breathing-Qwen is a controlled retrieval experiment around frozen **Qwen3-8B**. The motivating picture is simple: do not let every incoming cue rewrite the object you are trying to retrieve. Instead, alternate broad and sharp competition, measure what each cue leaves unexplained, and let that residue change how much the cue is trusted on the next cycle.

This repository is deliberately stricter than the metaphor. It does **not** claim that brains literally implement a softmax temperature, that recurrence replaces model size, or that Qwen is brain-like. It tests a small inference mechanism and keeps its failures.

## The mechanism

For cue-by-candidate score matrix `Z`, the external controller does:

```text
aggregate_t = weighted_average(Z, cue_trust_t)
p_t         = softmax(beta_t * aggregate_t)
residue_t   = disagreement(each cue, aggregate_t)
cue_trust   = robust_weight(residue_t)
```

The frozen breathing schedule is:

```text
beta  = [0.50, 1.75, 0.65, 2.00]
phase = [open, close, open, close]
```

There are two separate interventions:

1. **Breathing temperature** changes how broad or sharp the candidate distribution is.
2. **Residue-dependent trust** changes the evidence used by later cycles.

In this **external settling** implementation, breathing-only changes entropy but **does not change candidate ordering**. That is intentional: it is a matched control. The causal question in v0 is whether residue-dependent trust makes the next computation more robust to a contradictory cue.

## Gates

### Gate 0: FAIL — CPU associative-memory falsifier

The first official holdout was frozen before running: 256 clean/corrupt pairs, seed `1729`. The pass criterion was also frozen: the residue arm had to beat the best control on corrupted queries by at least **10 percentage points**, while losing no more than **2 points** on clean queries.

First official receipt:

| Arm | Clean accuracy | Corrupt accuracy |
|---|---:|---:|
| one-shot | 80.08% | 41.80% |
| fixed iteration | 80.08% | 41.80% |
| breathing-only | 80.08% | 41.80% |
| breathing + residue | **80.86%** | **47.27%** |

The residue arm improved corrupted accuracy by **+5.47 percentage points**, not the required +10. It recovered **13.4%** of misses made by the one-shot arm and slightly improved aggregate clean accuracy, but the predeclared gate still fails.

That failure is committed at `receipts/gate0_seed1729.json`. It will not be retuned away.

Run it yourself:

```bash
python scripts/run_toy.py --json
```

### Gate 1: NOT RUN — frozen Qwen3-8B external settling

Gate 1 uses Qwen only as a deterministic semantic evidence scorer. The model is frozen. No reasoning tokens are generated. Candidate answers are scored by full-string, length-normalized teacher-forced conditional log-likelihood.

The frozen benchmark contains **32 clean/corrupt pairs**. Exactly one cue is replaced by a plausible false cue in each corrupt condition. Its SHA-256 is:

```text
38322de99fc14af71966dad1ecdcaa5f972300b2c3814455912440408f1a0f96
```

The runner compares five arms on the same cached evidence:

- native Qwen with all cues in one prompt;
- per-cue one-shot aggregation;
- repeated fixed-temperature aggregation;
- breathing-only aggregation;
- breathing + residue-dependent trust.

It also reruns native all-cues scoring with cue order reversed, because position/order sensitivity is a known confound for this project.

### Gate 2: not implemented yet

The next stronger test is repeated inference over an unchanged prompt with no new generated tokens, while a controller carries span trust between passes.

### Gate 3: boundary only

`breathing_qwen/internal_attention.py` defines the intended selected-head score transform:

```text
scores' = beta * scores + span_bias
```

but only for explicitly selected heads. Every other head is an exact no-op. The adapter currently **refuses to install a live Qwen hook** because the installed Transformers/Qwen attention implementation has not been verified here. That is deliberate; Gate 3 stays unsupported until the hook can be tested against the real model implementation.

## Install

CPU / development tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not qwen" -v
```

Qwen run:

```bash
python -m pip install -e ".[test,qwen]"
```

Use an **exact model revision/commit**, not a floating branch name, for an official receipt.

A memory-conservative smoke run over two benchmark items:

```bash
python scripts/run_qwen.py \
  --revision <QWEN3_8B_COMMIT> \
  --limit 2 \
  --candidate-batch-size 1 \
  --max-memory-json '{"0":"6GiB","cpu":"6GiB"}' \
  --offload-folder .offload_qwen
```

If that completes, run the frozen 32-pair Gate 1 by removing `--limit 2`:

```bash
python scripts/run_qwen.py \
  --revision <QWEN3_8B_COMMIT> \
  --candidate-batch-size 1 \
  --max-memory-json '{"0":"6GiB","cpu":"6GiB"}' \
  --offload-folder .offload_qwen
```

`--max-memory-json` and disk offload are optional. Increase `--candidate-batch-size` only if memory allows it.

Every run writes a JSON receipt under `receipts/` containing the model revision, benchmark hash, controller constants, metrics, per-item predictions, settling traces, and environment metadata.

Summarize receipts without rerunning Qwen:

```bash
python scripts/summarize_receipts.py receipts/gate1_*.json
```

## What would count as interesting?

A positive Gate 1 would not mean "rhythmic brains explain LLMs." It would mean something narrower:

> Under deliberately contradictory cues, a frozen model's per-cue evidence contains useful support that a robust iterative controller can recover more reliably than matched one-shot/fixed/breathing controls.

A negative result is equally acceptable. If residue settling does not beat the controls, that is the result.

The stronger claim — that changing **internal** attention dynamics lets a frozen Qwen model change its preference without new tokens — belongs to Gate 3 and is not claimed here.

## Repository layout

```text
breathing_qwen/
  schedules.py          frozen beta schedule
  robust.py             robust cue weighting
  settling.py           four external settling arms
  toy_memory.py         Gate-0 CPU falsifier
  benchmark.py          paired benchmark schema
  qwen_score.py         teacher-forced Qwen candidate scoring
  receipts.py           Gate-1 runner, metrics and provenance
  internal_attention.py safe Gate-3 intervention boundary
benchmarks/v0.jsonl     frozen 32-pair benchmark
receipts/               committed official receipts
scripts/                runnable experiment CLIs
tests/                  CPU/unit tests; Qwen integration is opt-in
```

## Claim boundary

Supported now:

- the controller and benchmark are deterministic/tested;
- Gate 0 produced a measurable improvement but **failed its frozen pass criterion**;
- the Qwen Gate-1 harness is implemented and testable without downloading the model.

Not supported yet:

- that Gate 1 improves Qwen retrieval;
- that internal attention breathing works;
- that this explains biological rhythm;
- that a smaller recurrent model can replace a larger transformer.

The point of the repo is to make those boundaries hard to blur.
