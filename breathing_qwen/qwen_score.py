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
        import torch

        if len(input_ids) < 2:
            return np.empty(0, dtype=float)
        device = self.model.get_input_embeddings().weight.device
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits[0, :-1, :].float()
            targets = ids[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            observed = log_probs.gather(1, targets[:, None]).squeeze(1)
        return observed.detach().cpu().numpy().astype(float, copy=False)


class QwenCandidateScorer:
    def __init__(self, *, model_backend: SequenceLogprobBackend, tokenizer):
        self.model_backend = model_backend
        self.tokenizer = tokenizer

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
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        return cls(
            model_backend=_HFBackend(model, model_name=model_name, revision=revision),
            tokenizer=tokenizer,
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

        scores: list[float] = []
        for candidate in candidates:
            candidate_ids = list(self.tokenizer.encode(" " + str(candidate), add_special_tokens=False))
            if not candidate_ids:
                raise ValueError(f"candidate tokenization produced no tokens: {candidate!r}")
            full_ids = prompt_ids + candidate_ids
            observed = np.asarray(self.model_backend.observed_token_logprobs(full_ids), dtype=float)
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
