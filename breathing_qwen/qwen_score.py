from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EvidenceScorer(Protocol):
    def score_cue(self, cue: str, candidates: Sequence[str]) -> np.ndarray: ...


class SequenceLogprobBackend(Protocol):
    model_name: str
    revision: str | None

    def observed_token_logprobs(self, input_ids: Sequence[int]) -> np.ndarray: ...


class _HFBackend:
    def __init__(self, model, *, model_name: str, revision: str | None):
        self.model = model
        self.model_name = model_name
        self.revision = revision

    def observed_token_logprobs(self, input_ids: Sequence[int]) -> np.ndarray:
        return self.observed_token_logprobs_batch([input_ids])[0]

    def observed_token_logprobs_batch(self, sequences: Sequence[Sequence[int]]) -> list[np.ndarray]:
        import torch

        if not sequences:
            return []
        lengths = [len(ids) for ids in sequences]
        if any(length < 2 for length in lengths):
            raise ValueError("all scoring sequences need at least two tokens")
        max_len = max(lengths)
        device = self.model.get_input_embeddings().weight.device
        ids = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.long, device=device)
            mask[row, : len(seq)] = 1
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
        outputs: list[np.ndarray] = []
        for row, length in enumerate(lengths):
            step_logits = logits[row, : length - 1, :]
            targets = ids[row, 1:length]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            observed = log_probs.gather(1, targets[:, None]).squeeze(1)
            outputs.append(observed.detach().cpu().numpy().astype(float, copy=False))
        return outputs


class QwenCandidateScorer:
    def __init__(
        self,
        *,
        model_backend: SequenceLogprobBackend,
        tokenizer,
        candidate_batch_size: int | None = None,
    ):
        if candidate_batch_size is not None and candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be >= 1 or None")
        self.model_backend = model_backend
        self.tokenizer = tokenizer
        self.candidate_batch_size = candidate_batch_size

    @property
    def model_name(self) -> str:
        return self.model_backend.model_name

    @property
    def revision(self) -> str | None:
        return self.model_backend.revision

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        revision: str | None,
        device_map: str | dict = "auto",
        max_memory: dict | None = None,
        offload_folder: str | None = None,
        candidate_batch_size: int | None = 1,
    ) -> "QwenCandidateScorer":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "qwen optional dependencies are required; install `breathing-qwen[qwen]`"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        kwargs = {"revision": revision, "device_map": device_map, "dtype": "auto"}
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            kwargs["offload_folder"] = offload_folder
            kwargs["offload_state_dict"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        return cls(
            model_backend=_HFBackend(model, model_name=model_name, revision=revision),
            tokenizer=tokenizer,
            candidate_batch_size=candidate_batch_size,
        )

    def _format_prompt(self, cue: str, candidates: Sequence[str]) -> str:
        body = (
            f"Clue: {cue}\n"
            f"Candidates: {' | '.join(candidates)}\n"
            "Return exactly one candidate as the answer."
        )
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_template is None:
            return body + "\nAnswer:"
        messages = [{"role": "user", "content": body}]
        try:
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)

    def score_cue(self, cue: str, candidates: Sequence[str]) -> np.ndarray:
        if not candidates or any(not str(candidate).strip() for candidate in candidates):
            raise ValueError("candidate strings must be nonempty")
        prompt = self._format_prompt(cue, candidates)
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not prompt_ids:
            raise ValueError("prompt tokenization produced no tokens")

        candidate_token_ids: list[list[int]] = []
        sequences: list[list[int]] = []
        for candidate in candidates:
            ids = list(self.tokenizer.encode(" " + str(candidate), add_special_tokens=False))
            if not ids:
                raise ValueError(f"candidate tokenization produced no tokens: {candidate!r}")
            candidate_token_ids.append(ids)
            sequences.append(prompt_ids + ids)

        batch_method = getattr(self.model_backend, "observed_token_logprobs_batch", None)
        if batch_method is not None:
            batch_size = self.candidate_batch_size or len(sequences)
            observed_rows = []
            for start in range(0, len(sequences), batch_size):
                observed_rows.extend(batch_method(sequences[start : start + batch_size]))
        else:
            observed_rows = [self.model_backend.observed_token_logprobs(ids) for ids in sequences]
        if len(observed_rows) != len(sequences):
            raise ValueError("backend returned wrong batch size")

        scores: list[float] = []
        for full_ids, candidate_ids, observed_raw in zip(
            sequences, candidate_token_ids, observed_rows, strict=True
        ):
            observed = np.asarray(observed_raw, dtype=float)
            if observed.shape != (len(full_ids) - 1,):
                raise ValueError("backend returned wrong number of token log-probabilities")
            if not np.all(np.isfinite(observed)):
                raise ValueError("model token log-probabilities must be finite")
            start = len(prompt_ids) - 1
            candidate_logprobs = observed[start:]
            if len(candidate_logprobs) != len(candidate_ids):
                raise ValueError("candidate log-probability slice length mismatch")
            scores.append(float(candidate_logprobs.mean()))
        result = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("candidate scores must be finite")
        return result
