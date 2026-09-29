import sys
import types

import numpy as np
import pytest

from breathing_qwen.qwen_score import QwenCandidateScorer


class FakeTokenizer:
    def __init__(self):
        self.mapping = {
            "PROMPT": [10, 11],
            " alpha": [1, 9],
            " beta": [4],
            " double": [2, 2],
            " single": [2],
        }

    def encode(self, text, add_special_tokens=False):
        if text.startswith("Clue:"):
            return list(self.mapping["PROMPT"])
        return list(self.mapping[text])


class FakeBackend:
    model_name = "fake"
    revision = "unit-test"

    def observed_token_logprobs(self, input_ids):
        return np.array([-token / 10.0 for token in input_ids[1:]], dtype=float)


class NonFiniteBackend(FakeBackend):
    def observed_token_logprobs(self, input_ids):
        out = super().observed_token_logprobs(input_ids)
        out[-1] = np.nan
        return out


def make_scorer(backend=None):
    return QwenCandidateScorer(model_backend=backend or FakeBackend(), tokenizer=FakeTokenizer())


def test_scores_full_multi_token_answer_not_only_first_token():
    scores = make_scorer().score_cue("anything", ["alpha", "beta"])
    assert np.allclose(scores, [-0.5, -0.4])
    assert int(np.argmax(scores)) == 1


def test_scores_are_length_normalized():
    scores = make_scorer().score_cue("anything", ["double", "single"])
    assert np.allclose(scores, [-0.2, -0.2])


def test_candidate_order_is_preserved_and_repeat_is_deterministic():
    scorer = make_scorer()
    a = scorer.score_cue("anything", ["beta", "alpha"])
    b = scorer.score_cue("anything", ["beta", "alpha"])
    assert np.allclose(a, b)
    assert np.allclose(a, [-0.4, -0.5])


def test_nonfinite_model_scores_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        make_scorer(NonFiniteBackend()).score_cue("anything", ["alpha", "beta"])


def test_empty_candidate_is_rejected():
    with pytest.raises(ValueError, match="candidate"):
        make_scorer().score_cue("anything", [""])


def test_from_pretrained_fails_clearly_without_optional_dependencies(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="qwen"):
        QwenCandidateScorer.from_pretrained("Qwen/Qwen3-8B", revision="abc")


def test_from_pretrained_4bit_passes_quantization_config(monkeypatch):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured["quantization_kwargs"] = kwargs

    class FakeModel:
        def eval(self):
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["model_kwargs"] = kwargs
            return FakeModel()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["tokenizer_kwargs"] = kwargs
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    QwenCandidateScorer.from_pretrained(
        "Qwen/Qwen3-8B",
        revision="abc",
        load_4bit=True,
    )

    assert captured["model_name"] == "Qwen/Qwen3-8B"
    assert captured["model_kwargs"]["revision"] == "abc"
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["model_kwargs"]["quantization_config"].__class__ is FakeBitsAndBytesConfig
    assert captured["quantization_kwargs"]["load_in_4bit"] is True
    assert captured["quantization_kwargs"]["bnb_4bit_quant_type"] == "nf4"
    assert captured["quantization_kwargs"]["bnb_4bit_compute_dtype"] is fake_torch.bfloat16


class BatchBackend(FakeBackend):
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def observed_token_logprobs(self, input_ids):
        self.single_calls += 1
        return super().observed_token_logprobs(input_ids)

    def observed_token_logprobs_batch(self, sequences):
        self.batch_calls += 1
        return [FakeBackend.observed_token_logprobs(self, ids) for ids in sequences]


def test_candidate_sequences_use_one_batch_backend_call_when_available():
    backend = BatchBackend()
    scores = make_scorer(backend).score_cue("anything", ["alpha", "beta"])
    assert np.allclose(scores, [-0.5, -0.4])
    assert backend.batch_calls == 1
    assert backend.single_calls == 0


def test_candidate_batch_size_can_reduce_peak_memory():
    backend = BatchBackend()
    scorer = QwenCandidateScorer(
        model_backend=backend,
        tokenizer=FakeTokenizer(),
        candidate_batch_size=1,
    )
    scores = scorer.score_cue("anything", ["alpha", "beta"])
    assert np.allclose(scores, [-0.5, -0.4])
    assert backend.batch_calls == 2
    assert backend.single_calls == 0
