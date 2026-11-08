"""정답 polyline → YOLO-seg 학습 데이터(타일 이미지 + polygon 라벨) 변환."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..detectors.base import iter_tiles
from .dataset import ImageRecord, rasterize_polylines


def export_yolo_seg(records: dict[str, list[ImageRecord]], out_dir: str | Path,
                    tile: int = 1280, overlap: float = 0.2, thickness: int = 6,
                    keep_empty_ratio: float = 0.2) -> Path:
    """records: {"train": [...], "val": [...]} → out_dir/{images,labels}/{split}/ + data.yaml.

    전선이 없는 타일은 keep_empty_ratio 비율만 남긴다 (배경 학습용, 너무 많으면 불균형).
    """
    out = Path(out_dir)
    rng = np.random.default_rng(0)
    for split, recs in records.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            if not rec.exists:
                continue
            img = cv2.imread(str(rec.path))
            mask = rasterize_polylines(rec.annotation, thickness).astype(np.uint8)
            h, w = mask.shape
            for x0, y0, x1, y1 in iter_tiles(h, w, tile, overlap):
                tm = mask[y0:y1, x0:x1]
                if not tm.any() and rng.random() > keep_empty_ratio:
                    continue
                stem = f"{rec.frame}_{Path(rec.annotation.name).stem}_{x0}_{y0}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img[y0:y1, x0:x1])
                lines = []
                contours, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                th, tw = tm.shape
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    pts = cnt.reshape(-1, 2).astype(float) / [tw, th]
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.ravel()))
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: powerline\n"
    )
    return yaml
