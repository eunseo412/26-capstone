"""전선 검출 평가.

전선은 폭이 수 px밖에 안 되는 긴 선이라 bbox IoU나 마스크 IoU는 몇 px 어긋남에도 크게 떨어진다.
그래서 선 기반 허용오차 지표를 쓴다:

  * 픽셀 Precision = 예측 골격(skeleton) 픽셀 중 정답 선에서 tol px 이내인 비율
  * 픽셀 Recall    = 정답 선 픽셀 중 예측 마스크에서 tol px 이내인 비율
  * 후보 단위 TP/FP = 후보 골격의 match_ratio 이상이 정답 근처면 TP

후보 단위 판정은 "VLM이 어떤 오검출(FP)을 제거했고, 어떤 정검출(TP)을 잘못 지웠는가"를 세는 데 쓴다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..data.dataset import ImageRecord, rasterize_polylines
from ..types import Candidate


def distance_to(mask: np.ndarray) -> np.ndarray:
    """각 픽셀에서 mask의 가장 가까운 True 픽셀까지 거리."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


@dataclass
class PixelCounts:
    pred_total: int = 0
    pred_hit: int = 0
    gt_total: int = 0
    gt_hit: int = 0

    def __iadd__(self, o: "PixelCounts"):
        self.pred_total += o.pred_total
        self.pred_hit += o.pred_hit
        self.gt_total += o.gt_total
        self.gt_hit += o.gt_hit
        return self

    @property
    def precision(self) -> float:
        return self.pred_hit / self.pred_total if self.pred_total else 0.0

    @property
    def recall(self) -> float:
        return self.gt_hit / self.gt_total if self.gt_total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class ImageEval:
    image_id: str
    pixel: PixelCounts
    cand_tp: list[int] = field(default_factory=list)
    cand_fp: list[int] = field(default_factory=list)

    @property
    def status(self) -> str:
        """이미지 단위 결과: 정답 / 오검출 포함 / 미검출 포함."""
        if self.pixel.gt_total and self.pixel.recall < 0.5:
            return "missed"
        if self.cand_fp:
            return "false_positive"
        return "correct"


class Evaluator:
    def __init__(self, tolerance_px: float = 10, match_ratio: float = 0.5):
        self.tol = tolerance_px
        self.match_ratio = match_ratio

    def gt_line(self, rec: ImageRecord) -> np.ndarray:
        return rasterize_polylines(rec.annotation, thickness=1)

    def evaluate_image(self, rec: ImageRecord, candidates: list[Candidate],
                       keep_ids: set[int] | None = None) -> ImageEval:
        """keep_ids가 주어지면 그 후보만 최종 예측으로 본다 (융합 후 평가)."""
        h, w = rec.annotation.height, rec.annotation.width
        gt = self.gt_line(rec)
        gt_dist = distance_to(gt)

        kept = [c for c in candidates if keep_ids is None or c.cand_id in keep_ids]
        pred = np.zeros((h, w), dtype=bool)
        tp, fp = [], []
        for c in kept:
            x0, y0, x1, y1 = c.bbox
            pred[y0:y1, x0:x1] |= c.mask
        for c in candidates:
            if self.is_match(c, gt_dist):
                tp.append(c.cand_id)
            else:
                fp.append(c.cand_id)
        if keep_ids is not None:
            tp = [i for i in tp if i in keep_ids]
            fp = [i for i in fp if i in keep_ids]

        pred_skel = skeletonize(pred) if pred.any() else pred
        pred_dist = distance_to(pred)
        counts = PixelCounts(
            pred_total=int(pred_skel.sum()),
            pred_hit=int((pred_skel & (gt_dist <= self.tol)).sum()),
            gt_total=int(gt.sum()),
            gt_hit=int((gt & (pred_dist <= self.tol)).sum()),
        )
        return ImageEval(rec.image_id, counts, tp, fp)

    def is_match(self, cand: Candidate, gt_dist: np.ndarray) -> bool:
        skel = skeletonize(cand.mask) if cand.mask.any() else cand.mask
        n = int(skel.sum())
        if n == 0:
            return False
        x0, y0, x1, y1 = cand.bbox
        near = (gt_dist[y0:y1, x0:x1] <= self.tol) & skel
        return near.sum() / n >= self.match_ratio


@dataclass
class Summary:
    name: str
    pixel: PixelCounts
    n_images: int
    n_candidates_tp: int
    n_candidates_fp: int
    status_counts: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_images": self.n_images,
            "precision": round(self.pixel.precision, 4),
            "recall": round(self.pixel.recall, 4),
            "f1": round(self.pixel.f1, 4),
            "candidates_tp": self.n_candidates_tp,
            "candidates_fp": self.n_candidates_fp,
            "image_status": self.status_counts,
        }


def summarize(name: str, evals: list[ImageEval]) -> Summary:
    total = PixelCounts()
    for e in evals:
        total += e.pixel
    return Summary(
        name=name,
        pixel=total,
        n_images=len(evals),
        n_candidates_tp=sum(len(e.cand_tp) for e in evals),
        n_candidates_fp=sum(len(e.cand_fp) for e in evals),
        status_counts=dict(Counter(e.status for e in evals)),
    )
