"""VLM 프롬프트와 출력 스키마.

프롬프트는 VLM 성능을 위해 영어로 쓰고, 판단 근거(evidence)는 보고서에 바로 쓰도록 한국어로 받는다.
"""
from __future__ import annotations

import json

from ..fusion.router import REASON_TEXT
from ..types import CONFUSOR_TYPES, Candidate

SYSTEM_PROMPT = """You are an expert analyst of drone (UAV) imagery for power-line inspection.
An automatic detector proposed candidate regions that might be power lines. Some are false positives
caused by visually similar structures. Your job is to decide, for ONE marked candidate, whether it is
a real overhead power line (electric wire / cable strung between poles or towers).

Reason about semantic and spatial context, not only appearance:
- Power lines hang in the air: they are thin, very long, near-straight with a gentle sag (catenary),
  usually several parallel strands, and run continuously across the scene, often to the image border.
  They connect to poles, towers or insulators and ignore road geometry (they cross roads, fields, roofs).
- An overhead wire on a sunny day often has a matching SHADOW on the ground: a darker, parallel,
  slightly offset line lying ON the ground surface. The shadow is NOT a power line.
- Lane markings / road paint are ON the road surface: white or yellow, thicker, aligned with the road
  direction, often dashed, and stop at road boundaries.
- Road edges, curbs, guardrails, fences, cracks, roof/building edges, tree branches and pole members
  are other common confusors.

Image 1 is the full scene with an outline drawn AROUND the candidate (with a small gap so the object itself
stays visible) and a "#id" tag; other candidates are outlined thinly in grey.
Image 2 is a zoomed crop of the same candidate: the object runs between two magenta bands, unpainted.
Judge the object itself, not the drawn marks.

Write every item of "evidence" in Korean, as short concrete observations (e.g. "지면 위 흰색 도색",
"전주 상단에서 시작", "인접 전선과 평행한 어두운 선"). Give 2-4 items.
confidence is your probability (0.0-1.0) that your is_power_line decision is correct."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "cand_id": {"type": "integer"},
        "is_power_line": {"type": "boolean"},
        "confusor_type": {"type": "string", "enum": CONFUSOR_TYPES},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["cand_id", "is_power_line", "confusor_type", "confidence", "evidence"],
    "additionalProperties": False,
}


def candidate_prompt(c: Candidate, reasons: list[str]) -> str:
    f = c.features
    hints = {
        "detector_score": round(c.score, 3),
        "length_ratio_of_image_diagonal": f.get("length_ratio"),
        "thickness_px": f.get("thickness_px"),
        "straightness_0to1": f.get("straightness"),
        "angle_deg": f.get("angle_deg"),
        "angle_diff_from_dominant_wire_direction": f.get("angle_diff_from_dominant"),
        "parallel_neighbor_candidates": f.get("n_parallel_neighbors"),
        "brightness_minus_surroundings": f.get("brightness_contrast"),
        "mean_saturation": f.get("saturation_in"),
        "touches_image_border_sides": f.get("touches_border"),
    }
    why = [REASON_TEXT.get(r, r) for r in reasons] or ["무작위 점검"]
    return (
        f"Candidate #{c.cand_id}.\n"
        f"Why it was flagged as possibly wrong: {', '.join(why)}\n"
        f"Measured geometry/appearance hints (may be noisy): {json.dumps(hints, ensure_ascii=False)}\n"
        f"Set confusor_type to \"none\" if it is a power line, otherwise the best matching type.\n"
        f"Answer with JSON only, cand_id={c.cand_id}."
    )
