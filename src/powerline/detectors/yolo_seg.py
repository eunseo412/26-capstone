"""Baseline: YOLO segmentation 모델 (ultralytics).

학습 데이터는 scripts/prepare_yolo.py 로 만든다 (정답 polyline → 굵기를 준 polygon, 타일 단위).
bbox 검출이 아니라 seg를 쓰는 이유: 대각선 전선의 bbox는 대부분이 배경이라 평가·비교가 무의미하다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Detector, iter_tiles, merge_masks


class YoloSegDetector(Detector):
    name = "yolo"

    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg["weights"])

    def _predict_raw(self, image: np.ndarray):
        h, w = image.shape[:2]
        tcfg = self.cfg.get("tile", {})
        tiles = (list(iter_tiles(h, w, tcfg["size"], tcfg["overlap"]))
                 if tcfg.get("enabled") else [(0, 0, w, h)])
        found = []
        for x0, y0, x1, y1 in tiles:
            tile = image[y0:y1, x0:x1]
            res = self.model.predict(tile, conf=self.cfg["conf"], imgsz=self.cfg["imgsz"], verbose=False)[0]
            if res.masks is None:
                continue
            th, tw = tile.shape[:2]
            for poly, score in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
                m = np.zeros((th, tw), dtype=np.uint8)
                if len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
                full = np.zeros((h, w), dtype=bool)
                full[y0:y1, x0:x1] = m.astype(bool)
                found.append((full, float(score), "powerline"))
        return merge_masks(found)
