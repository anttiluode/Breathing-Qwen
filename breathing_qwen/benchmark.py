from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

V0_BENCHMARK_SHA256 = "38322de99fc14af71966dad1ecdcaa5f972300b2c3814455912440408f1a0f96"
V1_HOLDOUT_SHA256 = "c0d45ec644f4755f8bfb879457a9b1ae1311c20b4697d0aead9dfb83510dafed"


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    answer: str
    clean_cues: tuple[str, ...]
    corrupt_cues: tuple[str, ...]
    corrupt_index: int
    candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.answer:
            raise ValueError("id and answer are required")
        if self.candidates.count(self.answer) != 1:
            raise ValueError(f"{self.id}: answer must occur exactly once in candidates")
        if len(self.clean_cues) != len(self.corrupt_cues) or not self.clean_cues:
            raise ValueError(f"{self.id}: clean/corrupt cue lengths must match and be nonempty")
        changed = [
            i
            for i, pair in enumerate(zip(self.clean_cues, self.corrupt_cues, strict=True))
            if pair[0] != pair[1]
        ]
        if changed != [self.corrupt_index]:
            raise ValueError(f"{self.id}: exactly corrupt_index must differ")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"{self.id}: candidates must be unique")

    def inference_view(self, corrupt: bool) -> dict[str, Any]:
        cues = self.corrupt_cues if corrupt else self.clean_cues
        return {"id": self.id, "cues": list(cues), "candidates": list(self.candidates)}


def load_benchmark(path: Path | str) -> list[BenchmarkItem]:
    p = Path(path)
    items: list[BenchmarkItem] = []
    seen: set[str] = set()
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = BenchmarkItem(
                    id=str(raw["id"]),
                    answer=str(raw["answer"]),
                    clean_cues=tuple(str(x) for x in raw["clean_cues"]),
                    corrupt_cues=tuple(str(x) for x in raw["corrupt_cues"]),
                    corrupt_index=int(raw["corrupt_index"]),
                    candidates=tuple(str(x) for x in raw["candidates"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid benchmark line {line_no}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"duplicate benchmark id: {item.id}")
            seen.add(item.id)
            items.append(item)
    return items
