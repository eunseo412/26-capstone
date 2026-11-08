"""검출기 공통 인터페이스와 타일링 유틸."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Candidate
from .features import add_relational_features, compute_features


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def _predict_raw(self, image: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
        """RGB 이미지 → [(전체 해상도 bool 마스크, 점수, 프롬프트/클래스명)]."""

    def predict(self, image: np.ndarray, image_id: str) -> list[Candidate]:
        cands = []
        for i, (mask, score, prompt) in enumerate(self._predict_raw(image)):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            c = Candidate(
                image_id=image_id, cand_id=len(cands), source=self.name,
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                mask=mask[y0:y1, x0:x1].copy(), score=float(score), prompt=prompt,
            )
            c.features = compute_features(c, image)
            cands.append(c)
        add_relational_features(cands)
        return cands


def iter_tiles(h: int, w: int, size: int, overlap: float):
    """(x0, y0, x1, y1) 타일 좌표. 마지막 타일은 이미지 끝에 맞춘다."""
    step = max(1, int(size * (1 - overlap)))

    def starts(n):
        if n <= size:
            return [0]
        s = list(range(0, n - size, step))
        return s + [n - size]

    for y in starts(h):
        for x in starts(w):
            yield x, y, min(x + size, w), min(y + size, h)


def merge_masks(masks: list[tuple[np.ndarray, float, str]], iou_thr: float = 0.3):
    """타일 경계에서 중복된 마스크를 합친다 (겹침이 큰 것끼리 OR, 점수는 최대값)."""
    merged: list[list] = []
    for m, s, p in sorted(masks, key=lambda t: -t[1]):
        for g in merged:
            inter = np.logical_and(g[0], m).sum()
            small = min(g[0].sum(), m.sum())
            if small and inter / small > iou_thr:
                g[0] |= m
                g[1] = max(g[1], s)
                break
        else:
            merged.append([m.copy(), s, p])
    return [tuple(g) for g in merged]
