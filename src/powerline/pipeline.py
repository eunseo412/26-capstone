"""Stage 1~5 실행기.

outputs/<run_id>/
  candidates/<image>.json   Stage 1 후보 (재실행 시 재사용 → VLM 프롬프트만 바꿔 실험 가능)
  decisions/<image>.json    Stage 3~5 결과 (라우팅 이유, VLM 판정, 최종 판정)
  marked/<image>__<id>.jpg  VLM에 보낸 overview+crop (보고서 사례 이미지로 재사용)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from .data.dataset import ImageRecord
from .fusion import fuse, route
from .marking import draw_crop, draw_overview, side_by_side
from .types import Candidate, FinalDecision, Verdict

log = logging.getLogger(__name__)


def safe_name(image_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", image_id)


def save_candidates(path: Path, cands: list[Candidate]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in cands]))


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_dict(d) for d in json.loads(path.read_text())]


def save_decisions(path: Path, decisions: list[FinalDecision]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([d.to_dict() for d in decisions], ensure_ascii=False, indent=1))


def load_decisions(path: Path) -> list[FinalDecision]:
    return [FinalDecision.from_dict(d) for d in json.loads(path.read_text())]


class Pipeline:
    def __init__(self, cfg: dict, detector, vlm, run_dir: str | Path, use_vlm: bool = True):
        self.cfg = cfg
        self.detector = detector
        self.vlm = vlm
        self.run_dir = Path(run_dir)
        self.use_vlm = use_vlm

    def candidates_for(self, rec: ImageRecord, image: np.ndarray) -> list[Candidate]:
        path = self.run_dir / "candidates" / f"{safe_name(rec.image_id)}.json"
        if path.exists():
            return load_candidates(path)
        if self.detector is None:
            raise RuntimeError(f"후보 캐시가 없고 검출기도 없음: {path}")
        cands = self.detector.predict(image, rec.image_id)
        save_candidates(path, cands)
        return cands

    def run_image(self, rec: ImageRecord) -> tuple[list[Candidate], list[FinalDecision]]:
        image = rec.load_image()
        cands = self.candidates_for(rec, image)
        routes = route(cands, self.cfg)
        verdicts: dict[int, Verdict] = {}
        if self.use_vlm:
            mcfg = self.cfg["marking"]
            cmap = {c.cand_id: c for c in cands}
            for r in routes:
                if not r.send_to_vlm:
                    continue
                c = cmap[r.cand_id]
                overview = draw_overview(image, cands, mcfg["overview_max_side"], highlight={c.cand_id})
                crop = draw_crop(image, c, mcfg["crop_context"], mcfg["crop_max_side"])
                try:
                    verdicts[c.cand_id] = self.vlm.judge(overview, crop, c, r.reasons)
                except Exception as e:
                    log.error("VLM 실패 %s #%d: %s", rec.image_id, c.cand_id, e)
                    continue
                mpath = self.run_dir / "marked" / f"{safe_name(rec.image_id)}__{c.cand_id}.jpg"
                mpath.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mpath), cv2.cvtColor(side_by_side(overview, crop), cv2.COLOR_RGB2BGR))
        decisions = fuse(cands, routes, verdicts, self.cfg)
        save_decisions(self.run_dir / "decisions" / f"{safe_name(rec.image_id)}.json", decisions)
        return cands, decisions

    def run(self, records: list[ImageRecord]):
        n = 0
        for rec in records:
            if not rec.exists:
                log.warning("이미지 없음, 건너뜀: %s", rec.path)
                continue
            cands, dec = self.run_image(rec)
            n += 1
            log.info("[%d] %s: 후보 %d, VLM %d, 최종 유지 %d", n, rec.image_id, len(cands),
                     sum(d.verdict is not None for d in dec), sum(d.keep for d in dec))
