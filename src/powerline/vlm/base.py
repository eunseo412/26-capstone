"""VLM 백엔드 공통 인터페이스."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import numpy as np

from ..types import CONFUSOR_TYPES, Candidate, Verdict
from .prompts import candidate_prompt


class VLMBackend(ABC):
    name = "base"

    @abstractmethod
    def _ask(self, images: list[np.ndarray], prompt: str) -> str:
        """이미지(RGB) 목록 + 텍스트 → JSON 문자열."""

    def generate_text(self, prompt: str, schema: dict | None = None, system: str | None = None) -> str:
        """텍스트 전용 생성 (보고서 작성용). 지원하지 않는 백엔드는 NotImplementedError."""
        raise NotImplementedError

    def judge(self, overview: np.ndarray, crop: np.ndarray, cand: Candidate,
              reasons: list[str]) -> Verdict:
        raw = self._ask([overview, crop], candidate_prompt(cand, reasons))
        return parse_verdict(raw, cand.cand_id, self.name)


def parse_verdict(raw: str, cand_id: int, backend: str) -> Verdict:
    """JSON 응답 파싱. 로컬 모델은 코드블록/잡담을 섞기도 하므로 첫 {...}를 찾아 읽는다."""
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0) if m else raw)
        ctype = d.get("confusor_type", "other")
        return Verdict(
            cand_id=cand_id,
            is_power_line=bool(d["is_power_line"]),
            confusor_type=ctype if ctype in CONFUSOR_TYPES else "other",
            confidence=float(np.clip(float(d.get("confidence", 0.5)), 0, 1)),
            evidence=[str(e) for e in d.get("evidence", [])][:4],
            backend=backend, raw=raw,
        )
    except Exception as e:  # 파싱 실패 → 판단 보류(확신도 0)로 처리, 융합에서 점수 기준으로 결정됨
        return Verdict(cand_id, True, "other", 0.0, [f"응답 파싱 실패: {e}"], backend, raw)
