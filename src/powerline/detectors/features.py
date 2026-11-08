"""후보 마스크의 기하·외관 특징.

VLM 라우팅 규칙(Stage 3)과 VLM 프롬프트의 보조 정보(Stage 4)로 쓰인다.
전선 vs 혼동 객체를 가르는 단서:
  * 전선: 매우 길고 가늘며 거의 직선, 굵기 일정, 여러 가닥이 평행, 이미지 끝까지 이어짐
  * 차선: 밝고 채도 낮은 흰색/노란색, 상대적으로 굵음, 점선이면 조각남
  * 그림자: 주변보다 어두움, 전선과 평행하게 지면에 생김
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..types import Candidate


def compute_features(cand: Candidate, image: np.ndarray) -> dict:
    """image: RGB uint8 원본."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = cand.bbox
    m = cand.mask
    area = int(m.sum())
    if area == 0:
        return {"area": 0}

    ys, xs = np.nonzero(m)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)

    # 주성분 방향 / 직선성
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T) if len(pts) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    straightness = float(1 - evals[0] / evals[1]) if evals[1] > 0 else 0.0
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180)

    skel = skeletonize(m)
    length = int(skel.sum())
    thickness = area / max(length, 1)

    # 외관: 마스크 내부 vs 주변 띠
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
    v_in = float(hsv[..., 2][m].mean())
    v_out = float(hsv[..., 2][ring].mean()) if ring.any() else v_in
    s_in = float(hsv[..., 1][m].mean())

    border = 3
    touches = sum([x0 <= border, y0 <= border, x1 >= w - border, y1 >= h - border])

    diag = float(np.hypot(w, h))
    return {
        "area": area,
        "length_px": length,
        "length_ratio": round(length / diag, 4),
        "thickness_px": round(thickness, 2),
        "straightness": round(straightness, 4),
        "angle_deg": round(angle, 1),
        "brightness_in": round(v_in, 1),
        "brightness_contrast": round(v_in - v_out, 1),   # +: 주변보다 밝음, -: 어두움
        "saturation_in": round(s_in, 1),
        "touches_border": touches,
    }


def add_relational_features(cands: list[Candidate]) -> None:
    """후보들 사이 관계: 평행한 이웃 수, 지배 방향과의 차이."""
    angles = [c.features.get("angle_deg") for c in cands]
    lengths = [c.features.get("length_px", 0) for c in cands]
    valid = [(a, l) for a, l in zip(angles, lengths) if a is not None]
    if not valid:
        return
    # 길이 가중 방향 히스토그램으로 지배 방향 추정 (전선 다발 방향)
    hist = np.zeros(36)
    for a, l in valid:
        hist[int(a // 5) % 36] += l
    hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [1, 1, 1], "valid")
    dominant = float(np.argmax(hist) * 5 + 2.5)
    for c, a in zip(cands, angles):
        if a is None:
            continue
        c.features["dominant_angle_deg"] = round(dominant, 1)
        c.features["angle_diff_from_dominant"] = round(_angdiff(a, dominant), 1)
        c.features["n_parallel_neighbors"] = sum(
            1 for o, b in zip(cands, angles) if o is not c and b is not None and _angdiff(a, b) < 5
        )


def _angdiff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)
