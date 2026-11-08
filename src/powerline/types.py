"""파이프라인 단계 사이에서 주고받는 공통 데이터 타입.

각 Stage 결과는 JSONL로 저장/로드할 수 있어야 하므로 모든 타입에 to_dict/from_dict를 둔다.
마스크는 원본 해상도 전체가 아니라 bbox 영역만 잘라 RLE로 저장한다 (4K 이미지 대비 용량 절약).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# 혼동 유형 목록 (VLM 출력, 오류 분석, 보고서에서 공통 사용)
CONFUSOR_TYPES = [
    "none",            # 전선 맞음
    "lane_marking",    # 도로 차선/노면 표시
    "shadow",          # 전선·구조물의 그림자
    "road_edge",       # 도로 경계, 연석
    "guardrail",       # 가드레일, 펜스
    "crack",           # 노면 균열, 이음매
    "building_edge",   # 건물/지붕 모서리
    "branch",          # 나뭇가지
    "pole",            # 전주, 철탑 부재
    "other",
]


# ---------------------------------------------------------------- RLE
def mask_to_rle(mask: np.ndarray) -> dict:
    """bool 2D 배열 → {'size': [h, w], 'counts': [...]} (row-major, 0부터 시작하는 run)."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return {"size": list(mask.shape), "counts": runs}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    out = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for run in rle["counts"]:
        if val:
            out[pos:pos + run] = True
        pos += run
        val = not val
    return out.reshape(h, w)


# ---------------------------------------------------------------- Candidate
@dataclass
class Candidate:
    """검출기가 낸 전선 후보 1개."""
    image_id: str
    cand_id: int
    source: str                       # 후보를 만든 사전학습 모델 이름 (예: "langsam")
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 (원본 좌표, x1/y1 미포함)
    mask: np.ndarray                  # bbox 크기의 bool 마스크
    score: float
    prompt: str = ""
    features: dict = field(default_factory=dict)

    def full_mask(self, height: int, width: int) -> np.ndarray:
        out = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = self.bbox
        out[y0:y1, x0:x1] = self.mask
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mask"] = mask_to_rle(self.mask)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        d = dict(d)
        d["mask"] = rle_to_mask(d["mask"])
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


# ---------------------------------------------------------------- Routing
@dataclass
class RouteDecision:
    """Stage 3: 이 후보를 VLM에 보낼지와 그 이유."""
    cand_id: int
    send_to_vlm: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- Verdict
@dataclass
class Verdict:
    """Stage 4: VLM 판정."""
    cand_id: int
    is_power_line: bool
    confusor_type: str
    confidence: float
    evidence: list[str]
    backend: str = ""
    raw: str = ""                     # 디버깅용 원문 응답

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(**d)


# ---------------------------------------------------------------- Final
@dataclass
class FinalDecision:
    """Stage 5: 융합 후 최종 판정."""
    image_id: str
    cand_id: int
    keep: bool
    rule: str                          # 어떤 규칙으로 결정됐는지 (보고서 근거)
    score: float
    verdict: Optional[Verdict] = None
    route_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FinalDecision":
        d = dict(d)
        if d.get("verdict"):
            d["verdict"] = Verdict.from_dict(d["verdict"])
        return cls(**d)
