# Gate 1B: residue-guided native requery

Status: **frozen before the first Gate 1B run**.

This is a post-hoc development experiment prompted by Gate 1-Q4. It uses the same 32-item benchmark and therefore cannot count as independent confirmation. A positive result only justifies a held-out follow-up benchmark.

## Question

Can the residue controller be used as a **corruption locator** while leaving final inference to Qwen's intact joint-conditioning path?

Gate 1 showed a strong asymmetry: per-cue decomposition damaged answer quality, while residue reweighting often identified the deliberately corrupted cue. Gate 1B therefore separates diagnosis from reconstruction.

## Frozen procedure

For each benchmark item, evaluate clean and corrupt conditions with the same candidate order.

1. **native_all_cues**: score all clues jointly.
2. **random_remove**: remove one deterministic pseudo-random clue, then score the remaining clues jointly. The index is derived from SHA-256 of `gate1b-v0:<item.id>` and is identical for clean/corrupt views.
3. **residue_guided**: score clues individually, run the frozen Gate-1 `breathing_residue` controller, remove the single clue with the lowest final trust, then score the remaining clues jointly.
4. **oracle_remove**: remove `corrupt_index`, then score the remaining clues jointly. On clean items this removes the paired true clue; on corrupt items it removes the known injected corruption. This is an upper-bound control, not a deployable method.
5. **exhaustive_leave_one_out**: score every one-clue-removed joint prompt, softmax each candidate score vector, average those probability vectors, and choose the largest mean probability. This measures neighborhood stability without using labels.

All candidate scores are the existing deterministic, length-normalized teacher-forced conditional log-likelihoods with thinking disabled.

The residue detector uses the frozen Gate-1 schedule and robust constants unchanged:

- beta schedule: `[0.50, 1.75, 0.65, 2.00]`
- phases: `[open, close, open, close]`
- Huber delta: `0.20`
- trust floor: `0.05`

No threshold or schedule tuning is allowed after the run starts.

## Frozen decision rule

Gate 1B passes only if all conditions hold on the exact 32-item v0 benchmark:

- residue-guided corrupt accuracy is at least **2/32 (0.0625)** above native all-cues;
- residue-guided corrupt accuracy is at least **2/32 (0.0625)** above deterministic random removal;
- residue-guided clean accuracy loses at most **1/32 (0.03125)** versus native all-cues;
- on corrupt items, `(native miss -> guided recovery) - (native correct -> guided new error) >= 2`.

The locator accuracy (`argmin(final cue trust) == corrupt_index`) is reported diagnostically but is **not** a pass criterion because Gate 1-Q4 already exposed that signal on this same benchmark.

## Provenance boundary

A 4-bit NF4 run must identify itself as such in the receipt. It must not be silently reported as the original full-precision Gate 1. Gate 1B results from this 32-item benchmark are development-set evidence only.
