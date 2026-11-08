"""규칙 기반 모의 백엔드 — GPU/API 없이 파이프라인 전체 흐름을 점검하기 위한 용도.

실제 VLM이 아니므로 이 백엔드로 나온 수치는 성능 결과로 쓰면 안 된다.
"""
from __future__ import annotations

import json

from .base import VLMBackend


class MockBackend(VLMBackend):
    name = "mock"

    def __init__(self, cfg: dict | None = None):
        self._current = None

    def judge(self, overview, crop, cand, reasons):
        self._current = (cand, reasons)
        return super().judge(overview, crop, cand, reasons)

    def _ask(self, images, prompt):
        cand, reasons = self._current
        f = cand.features
        if "lane_marking" in reasons:
            d = (False, "lane_marking", 0.7, ["[mock] 밝고 채도 낮은 선"])
        elif f.get("n_parallel_neighbors", 0) >= 1 and f.get("straightness", 0) > 0.95:
            d = (True, "none", 0.7, ["[mock] 평행한 이웃 전선 존재", "[mock] 직선성 높음"])
        elif "shadow" in reasons:
            d = (False, "shadow", 0.6, ["[mock] 주변보다 어두운 선"])
        else:
            d = (cand.score >= 0.3, "other" if cand.score < 0.3 else "none", 0.5, ["[mock] 점수 기준"])
        return json.dumps({"cand_id": cand.cand_id, "is_power_line": d[0], "confusor_type": d[1],
                           "confidence": d[2], "evidence": d[3]}, ensure_ascii=False)
