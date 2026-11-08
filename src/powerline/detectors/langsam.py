"""Stage 1: LangSAM(GroundingDINO + SAM) 기반 전선 후보 생성.

설치: pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
목표는 recall: 임계값을 낮게 두고 후보를 많이 뽑은 뒤, 걸러내는 일은 VLM이 한다.
전선은 4K 이미지에서 몇 px 굵기라 원본 그대로 넣으면 GroundingDINO가 거의 못 찾는다 → 타일링.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Detector, iter_tiles, merge_masks


class LangSAMDetector(Detector):
    name = "langsam"

    def __init__(self, cfg: dict):
        from lang_sam import LangSAM  # 무거운 의존성은 사용할 때만 import

        self.cfg = cfg
        self.model = LangSAM()

    def _predict_tile(self, tile: np.ndarray, prompt: str):
        pil = Image.fromarray(tile)
        out = self.model.predict(
            [pil], [prompt],
            box_threshold=self.cfg["box_threshold"],
            text_threshold=self.cfg["text_threshold"],
        )
        res = out[0]
        masks = np.asarray(res.get("masks", []))
        scores = np.asarray(res.get("scores", []), dtype=float).reshape(-1)
        if masks.size == 0:
            return []
        if masks.ndim == 4:
            masks = masks[:, 0]
        return [(m > 0.5, float(s)) for m, s in zip(masks, scores)]

    def _predict_raw(self, image: np.ndarray):
        h, w = image.shape[:2]
        tcfg = self.cfg.get("tile", {})
        tiles = (list(iter_tiles(h, w, tcfg["size"], tcfg["overlap"]))
                 if tcfg.get("enabled") else [(0, 0, w, h)])
        found = []
        for prompt in self.cfg["text_prompts"]:
            for x0, y0, x1, y1 in tiles:
                for m, s in self._predict_tile(image[y0:y1, x0:x1], prompt):
                    if m.sum() < self.cfg.get("min_area_px", 0):
                        continue
                    full = np.zeros((h, w), dtype=bool)
                    full[y0:y1, x0:x1] = m
                    found.append((full, s, prompt))
        return merge_masks(found)
