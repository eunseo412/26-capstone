"""비교군별 평가와 VLM 효과 분석 → metrics.json (보고서 생성 입력)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ..data.dataset import ImageRecord
from ..pipeline import load_candidates, load_decisions, safe_name
from .metrics import Evaluator, distance_to, summarize


def evaluate_run(run_dir: str | Path, records: list[ImageRecord], cfg: dict) -> dict:
    """run_dir: LangSAM(+VLM) 결과. 학습이 없으므로 같은 후보 집합에서 세 방식을 비교한다.

    비교군:
      detector_all    : LangSAM이 낸 후보 전부 (recall 상한 참고용)
      detector_only   : LangSAM 단독 = 점수 keep_score 이상만 전선으로 봄 (일반적인 OD 사용법, 기준 방식)
      detector_vlm    : LangSAM + VLM 최종 판정 (제안 파이프라인)
    """
    ev = Evaluator(cfg["eval"]["tolerance_px"], cfg["eval"]["instance_match_ratio"])
    keep_score = cfg["fusion"]["keep_score"]
    run_dir = Path(run_dir)
    groups = defaultdict(list)
    vlm_effect = Counter()
    by_confusor = defaultdict(Counter)
    cases = []
    per_image = []

    for rec in records:
        name = safe_name(rec.image_id)
        cpath, dpath = run_dir / "candidates" / f"{name}.json", run_dir / "decisions" / f"{name}.json"
        if not (cpath.exists() and dpath.exists()):
            continue
        cands, decs = load_candidates(cpath), load_decisions(dpath)
        dmap = {d.cand_id: d for d in decs}

        groups["detector_all"].append(ev.evaluate_image(rec, cands))
        only = ev.evaluate_image(rec, cands, {c.cand_id for c in cands if c.score >= keep_score})
        full = ev.evaluate_image(rec, cands, {d.cand_id for d in decs if d.keep})
        groups["detector_only"].append(only)
        groups["detector_vlm"].append(full)
        per_image.append({"image_id": rec.image_id, "before": only.status, "after": full.status})

        # VLM 효과: VLM이 본 후보들의 판정이 정답과 맞았는지
        gt_dist = distance_to(ev.gt_line(rec))
        gt_tp = {c.cand_id for c in cands if ev.is_match(c, gt_dist)}
        for d in decs:
            if d.verdict is None:
                continue
            truth = d.cand_id in gt_tp
            before = d.score >= keep_score
            if before == d.keep:
                key = "unchanged"
            elif not d.keep and not truth:
                key = "fp_removed"          # 좋음: 오검출 제거
            elif not d.keep and truth:
                key = "tp_removed"          # 나쁨: 맞는 전선 제거
            elif d.keep and truth:
                key = "tp_recovered"        # 좋음: 점수 낮던 전선 살림
            else:
                key = "fp_added"            # 나쁨
            vlm_effect[key] += 1
            vlm_effect["vlm_correct" if d.verdict.is_power_line == truth else "vlm_wrong"] += 1
            if not truth:
                by_confusor[d.verdict.confusor_type]["removed" if not d.keep else "kept"] += 1
            cases.append({
                "image_id": rec.image_id, "cand_id": d.cand_id, "truth_is_power_line": truth,
                "score": round(d.score, 3), "route_reasons": d.route_reasons,
                "vlm_is_power_line": d.verdict.is_power_line, "confusor_type": d.verdict.confusor_type,
                "confidence": d.verdict.confidence, "evidence": d.verdict.evidence,
                "effect": key, "keep": d.keep,
                "marked_image": str(run_dir / "marked" / f"{name}__{d.cand_id}.jpg"),
            })

    return {
        "n_images": len(per_image),
        "tolerance_px": cfg["eval"]["tolerance_px"],
        "summaries": {k: summarize(k, v).to_dict() for k, v in groups.items()},
        "vlm_effect": dict(vlm_effect),
        "fp_by_confusor": {k: dict(v) for k, v in by_confusor.items()},
        "image_status_change": dict(Counter(f"{p['before']}->{p['after']}" for p in per_image)),
        "cases": cases,
    }

