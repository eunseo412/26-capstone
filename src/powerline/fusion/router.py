"""Stage 3: '오답 가능성이 높은' 후보만 골라 VLM에 보낸다.

선별 이유(reasons)는 그대로 VLM 프롬프트와 보고서의 판단 근거에 들어간다.
"""
from __future__ import annotations

from ..types import Candidate, RouteDecision

REASON_TEXT = {
    "uncertain_score": "검출 점수가 애매한 구간",
    "lane_marking": "밝고 채도 낮은 직선(차선 의심)",
    "shadow": "주변보다 어두운 선(그림자 의심)",
    "orientation_outlier": "다른 전선 후보들과 방향이 다름",
    "short_segment": "짧은 조각",
}


def route(cands: list[Candidate], cfg: dict) -> list[RouteDecision]:
    rcfg = cfg["router"]
    rules = rcfg.get("suspect_rules", {})
    out = []
    for c in cands:
        f = c.features
        reasons = []
        if rcfg["score_low"] <= c.score < rcfg["score_high"]:
            reasons.append("uncertain_score")
        if rules.get("lane_marking") and f.get("brightness_contrast", 0) > 25 and f.get("saturation_in", 255) < 80 \
                and f.get("thickness_px", 0) > 4:
            reasons.append("lane_marking")
        if rules.get("shadow") and f.get("brightness_contrast", 0) < -25:
            reasons.append("shadow")
        if rules.get("orientation_outlier") and f.get("angle_diff_from_dominant", 0) > 20:
            reasons.append("orientation_outlier")
        if rules.get("short_segment") and f.get("length_ratio", 1) < 0.05:
            reasons.append("short_segment")
        out.append(RouteDecision(c.cand_id, bool(reasons), reasons))
    return out
