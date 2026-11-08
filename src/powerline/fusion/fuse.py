"""Stage 5: 검출 점수 + VLM 판정 → 최종 판정. 규칙과 그 이름(rule)을 함께 남긴다."""
from __future__ import annotations

from ..types import Candidate, FinalDecision, RouteDecision, Verdict


def fuse(cands: list[Candidate], routes: list[RouteDecision], verdicts: dict[int, Verdict],
         cfg: dict) -> list[FinalDecision]:
    fcfg = cfg["fusion"]
    rmap = {r.cand_id: r for r in routes}
    out = []
    for c in cands:
        r = rmap.get(c.cand_id)
        v = verdicts.get(c.cand_id)
        if v is None:
            keep = c.score >= fcfg["keep_score"]
            rule = "score_only_keep" if keep else "score_only_drop"
        elif not v.is_power_line and v.confidence >= fcfg["reject_confidence"]:
            keep, rule = False, "vlm_reject"
        elif v.is_power_line and v.confidence >= fcfg["accept_confidence"]:
            keep, rule = True, "vlm_accept"
        else:
            keep = c.score >= fcfg["keep_score"]
            rule = "vlm_unsure_score_keep" if keep else "vlm_unsure_score_drop"
        out.append(FinalDecision(
            image_id=c.image_id, cand_id=c.cand_id, keep=keep, rule=rule, score=c.score,
            verdict=v, route_reasons=r.reasons if r else [],
        ))
    return out
