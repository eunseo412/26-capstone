"""Stage 2: Set-of-Mark 방식 마킹.

VLM에 두 종류의 이미지를 준다.
  1. overview: 원본 전체에 후보별 색 윤곽 + 번호 태그 → 장면 맥락 (도로, 전주, 그림자 방향)
  2. crop: 후보 주변을 확대한 이미지 → 국소 디테일 (색, 굵기, 지면 위인지 공중인지)
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from ..types import Candidate

GAP_PX = 6        # overview에서 윤곽과 전선 사이 간격
CROP_GAP_PX = 8   # crop에서 표시 띠와 전선 사이 간격


def palette(n: int) -> list[tuple[int, int, int]]:
    cols = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.9, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


def _resize_max(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def _label_point(c: Candidate) -> tuple[int, int]:
    ys, xs = np.nonzero(c.mask)
    k = len(xs) // 2
    order = np.argsort(xs + ys)
    return int(xs[order[k]] + c.bbox[0]), int(ys[order[k]] + c.bbox[1])


def _draw_tag(img, text, xy, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6 * scale, max(1, int(2 * scale))
    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
    x, y = xy
    x = int(np.clip(x, 0, img.shape[1] - tw - 6))
    y = int(np.clip(y, tht + 6, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - tht - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, fs, (0, 0, 0), th, cv2.LINE_AA)


def draw_overview(image: np.ndarray, cands: list[Candidate], max_side: int = 1568,
                  highlight: set[int] | None = None) -> np.ndarray:
    """highlight에 든 후보는 진하게, 나머지는 회색 얇은 선으로 (맥락용)."""
    small, s = _resize_max(image, max_side)
    out = small.copy()
    cols = palette(len(cands))
    lw = max(2, int(round(3 * small.shape[1] / 1568)))
    for c in cands:
        hl = highlight is None or c.cand_id in highlight
        col = cols[c.cand_id % len(cols)] if hl else (160, 160, 160)
        # bbox 영역만 축소해서 윤곽을 구한다 (4K 전체 마스크를 매번 만들지 않도록)
        x0, y0, x1, y1 = c.bbox
        sx0, sy0 = int(x0 * s), int(y0 * s)
        sw, sh = max(1, int(round(x1 * s)) - sx0), max(1, int(round(y1 * s)) - sy0)
        m = cv2.resize(c.mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        # 전선은 몇 px라 윤곽을 바로 위에 그리면 VLM이 전선 자체를 못 본다 → 간격을 두고 바깥에 그림
        gap = 2 * GAP_PX + 1 if hl else 3
        pad = gap
        m = cv2.copyMakeBorder((m > 0).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        m = cv2.dilate(m, np.ones((gap, gap), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                       offset=(sx0 - pad, sy0 - pad))
        cv2.drawContours(out, contours, -1, col, 2 if hl else 1)
        if hl:
            x, y = _label_point(c)
            _draw_tag(out, f"#{c.cand_id}", (int(x * s) + 4, int(y * s) - 4), col, small.shape[1] / 1568)
    return out


def draw_crop(image: np.ndarray, cand: Candidate, context: float = 1.6,
              max_side: int = 768) -> np.ndarray:
    """후보 bbox를 context배 넓혀 자르고 윤곽을 그린다. 선이 가려지지 않게 반투명 윤곽만."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = cand.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 64) * context / 2
    X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
    X1, Y1 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = image[Y0:Y1, X0:X1].copy()
    m = cand.full_mask(h, w)[Y0:Y1, X0:X1].astype(np.uint8)
    crop, s = _resize_max(crop, max_side)
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 전선 양옆으로 간격을 둔 띠만 칠한다 (전선 픽셀은 원본 그대로 보이게)
    inner = cv2.dilate(m, np.ones((2 * CROP_GAP_PX + 1,) * 2, np.uint8))
    outer = cv2.dilate(m, np.ones((2 * (CROP_GAP_PX + 3) + 1,) * 2, np.uint8))
    band = (outer - inner).astype(bool)
    out = crop.copy()
    out[band] = (0.35 * out[band] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    _draw_tag(out, f"#{cand.cand_id}", (6, 30), (255, 0, 255), 1.0)
    return out


def side_by_side(*imgs: np.ndarray, height: int = 360) -> np.ndarray:
    """보고서 썸네일용 가로 이어붙이기."""
    rs = [cv2.resize(i, (int(i.shape[1] * height / i.shape[0]), height)) for i in imgs]
    return np.concatenate(rs, axis=1)
