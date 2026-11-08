"""실행 결과 진단용 시각화 + 후보별 표.

이미지별로 정답과 후보를 한 장에 겹쳐 그린다:
  초록 얇은 선  : 정답 전선
  파랑          : 유지 + 정답 (맞게 남김)
  주황          : 유지 + 오답 (남은 오검출)
  회색          : 제거 + 오답 (맞게 지움)
  빨강          : 제거 + 정답 (전선을 잘못 지움)
VLM이 틀린 사례의 마킹 이미지는 vlm_wrong/ 으로 복사한다.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..data.dataset import ImageRecord, rasterize_polylines
from ..pipeline import load_candidates, load_decisions, safe_name
from .metrics import Evaluator, distance_to

COLORS = {  # RGB
    ("keep", True): (40, 110, 255),
    ("keep", False): (255, 150, 0),
    ("drop", False): (150, 150, 150),
    ("drop", True): (255, 30, 30),
}
LEGEND = [("GT", (0, 220, 0)), ("keep+wire", COLORS[("keep", True)]), ("keep+NOT wire", COLORS[("keep", False)]),
          ("removed+NOT wire", COLORS[("drop", False)]), ("removed+wire", COLORS[("drop", True)])]


def _scale(img: np.ndarray, width: int):
    s = width / img.shape[1]
    return cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def visualize_run(run_dir: str | Path, records: list[ImageRecord], cfg: dict, out_dir: str | Path,
                  width: int = 1920) -> Path:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "vlm_wrong").mkdir(parents=True, exist_ok=True)
    ev = Evaluator(cfg["eval"]["tolerance_px"], cfg["eval"]["instance_match_ratio"])
    rows = []
    for rec in records:
        name = safe_name(rec.image_id)
        cpath, dpath = run_dir / "candidates" / f"{name}.json", run_dir / "decisions" / f"{name}.json"
        if not (cpath.exists() and dpath.exists()):
            continue
        cands = load_candidates(cpath)
        dmap = {d.cand_id: d for d in load_decisions(dpath)}
        gt_dist = distance_to(ev.gt_line(rec))
        img = rec.load_image()
        h, w = img.shape[:2]
        small, s = _scale(img, width)
        layer = small.copy()

        gt = rasterize_polylines(rec.annotation, thickness=max(2, int(2 / s)))
        gt_small = cv2.resize(gt.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        layer[gt_small.astype(bool)] = (0, 220, 0)

        for c in sorted(cands, key=lambda c: -c.mask.sum()):   # 큰 것부터 그려 작은 것이 위에 보이게
            d = dmap.get(c.cand_id)
            truth = ev.is_match(c, gt_dist)
            keep = bool(d and d.keep)
            col = COLORS[("keep" if keep else "drop", truth)]
            x0, y0, x1, y1 = c.bbox
            sx0, sy0 = int(x0 * s), int(y0 * s)
            sw, sh = max(1, int(round(x1 * s)) - sx0), max(1, int(round(y1 * s)) - sy0)
            m = cv2.resize(c.mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0
            region = layer[sy0:sy0 + sh, sx0:sx0 + sw]
            region[m[:region.shape[0], :region.shape[1]]] = col
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                       offset=(sx0, sy0))
            cv2.drawContours(small, cnts, -1, col, 2)
            ys, xs = np.nonzero(m)
            if len(xs):
                k = len(xs) // 2
                cv2.putText(small, f"#{c.cand_id}", (int(xs[k] + sx0), int(ys[k] + sy0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

            v = d.verdict if d else None
            f = c.features
            rows.append({
                "image_id": rec.image_id, "cand_id": c.cand_id, "score": round(c.score, 3),
                "prompt": c.prompt, "truth_is_wire": truth, "keep": keep, "rule": d.rule if d else "",
                "area_px": f.get("area"), "area_ratio": round(f.get("area", 0) / (h * w), 4),
                "length_px": f.get("length_px"), "thickness_px": f.get("thickness_px"),
                "straightness": f.get("straightness"), "brightness_contrast": f.get("brightness_contrast"),
                "route_reasons": "|".join(d.route_reasons) if d else "",
                "vlm_is_wire": "" if v is None else v.is_power_line,
                "vlm_type": "" if v is None else v.confusor_type,
                "vlm_conf": "" if v is None else v.confidence,
                "vlm_evidence": "" if v is None else " / ".join(v.evidence),
            })
            if v is not None and v.is_power_line != truth:
                src = run_dir / "marked" / f"{name}__{c.cand_id}.jpg"
                if src.exists():
                    tag = "said_wire" if v.is_power_line else "said_not_wire"
                    shutil.copy(src, out_dir / "vlm_wrong" / f"{name}__{c.cand_id}__{tag}.jpg")

        out = cv2.addWeighted(layer, 0.45, small, 0.55, 0)
        y = 30
        for label, col in LEGEND:
            cv2.rectangle(out, (10, y - 18), (30, y + 2), col, -1)
            cv2.putText(out, label, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, label, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            y += 28
        cv2.imwrite(str(out_dir / "images" / f"{Path(name).stem}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    csv_path = out_dir / "candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        if rows:
            wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return out_dir
