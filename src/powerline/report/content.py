"""Stage 7-1: 보고서 '내용' 생성.

역할 분리 원칙
  * 숫자(성능표)는 metrics.json에서 코드가 직접 채운다 → LLM이 수치를 지어낼 여지 없음
  * LLM은 서술 문장만 쓴다. 필드별 글자 수 상한을 둬서 1페이지 분량을 보장
  * LLM 백엔드가 없으면(none) 같은 필드를 템플릿 문장으로 채운다
"""
from __future__ import annotations

import json
import logging

from ..fusion.router import REASON_TEXT

log = logging.getLogger(__name__)

CONFUSOR_KO = {
    "none": "전선", "lane_marking": "차선", "shadow": "그림자", "road_edge": "도로 경계",
    "guardrail": "가드레일", "crack": "노면 균열", "building_edge": "건물 모서리",
    "branch": "나뭇가지", "pole": "전주 부재", "other": "기타",
}
GROUP_KO = {"baseline": "Baseline OD", "detector_only": "LangSAM 단독", "detector_vlm": "LangSAM + VLM(제안)"}

LIMITS = {"overview": 160, "pipeline": 260, "performance": 220, "case": 130, "conclusion": 220, "action": 60}

CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "pipeline": {"type": "string"},
        "performance": {"type": "string"},
        "case_explanations": {"type": "array", "items": {"type": "string"}},
        "conclusion": {"type": "string"},
        "action_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overview", "pipeline", "performance", "case_explanations", "conclusion", "action_items"],
    "additionalProperties": False,
}


def select_cases(metrics: dict, k: int = 2) -> list[dict]:
    """대표 사례: VLM이 오검출을 제거한 사례(확신도 순) + 실패/회복 사례 1건(있으면)."""
    cases = metrics.get("cases", [])
    good = sorted([c for c in cases if c["effect"] == "fp_removed"], key=lambda c: -c["confidence"])
    other = [c for c in cases if c["effect"] in ("tp_recovered", "tp_removed", "fp_added")]
    picked = good[: max(1, k - (1 if other else 0))] + other[:1]
    if len(picked) < k:
        rest = [c for c in cases if c not in picked]
        picked += sorted(rest, key=lambda c: -c["confidence"])[: k - len(picked)]
    return picked[:k]


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _facts(metrics: dict, cases: list[dict]) -> dict:
    """LLM에 줄 사실 요약 (원문 metrics 전체 대신 필요한 것만)."""
    return {
        "n_images": metrics["n_images"],
        "tolerance_px": metrics["tolerance_px"],
        "summaries": {GROUP_KO.get(k, k): v for k, v in metrics["summaries"].items()},
        "vlm_effect": metrics["vlm_effect"],
        "fp_by_confusor": {CONFUSOR_KO.get(k, k): v for k, v in metrics["fp_by_confusor"].items()},
        "cases": [{
            "후보": f"{c['image_id']} #{c['cand_id']}",
            "선별 이유": [REASON_TEXT.get(r, r) for r in c["route_reasons"]],
            "VLM 판정": "전선" if c["vlm_is_power_line"] else CONFUSOR_KO.get(c["confusor_type"], c["confusor_type"]),
            "확신도": c["confidence"], "VLM 근거": c["evidence"],
            "정답": "전선" if c["truth_is_power_line"] else "전선 아님", "결과": c["effect"],
        } for c in cases],
    }


PROMPT = """아래는 드론 영상 전선 검출 파이프라인(LangSAM 후보 생성 → 번호 마킹 → 오답 가능성 높은 후보만 VLM 판정 → 결정 융합)의
평가 결과입니다. 담당자가 확인할 1페이지 공식 보고서의 서술 부분을 작성하세요.

규칙:
- 개조식 공문서 문체(~함, ~임)로 간결하게. 과장 금지, 주어진 수치만 사용하고 새 수치를 만들지 말 것.
- 필드별 최대 글자 수: overview {overview}, pipeline {pipeline}, performance {performance},
  case_explanations 각 {case}, conclusion {conclusion}, action_items 각 {action} (최대 3개).
- case_explanations는 cases 순서대로, VLM이 무엇을 근거로 어떻게 판단했고 그 판단이 정답과 맞았는지 서술.
- performance는 제안 방식과 비교군의 Precision/Recall/F1 차이와 혼동 유형별 오검출 제거를 해석.
- 개선이 없거나 악화된 지표가 있으면 그대로 밝힐 것.

평가 결과(JSON):
{facts}"""


def write_content(metrics: dict, cases: list[dict], llm=None) -> dict:
    facts = _facts(metrics, cases)
    content = None
    if llm is not None:
        try:
            raw = llm.generate_text(
                PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1), **LIMITS),
                schema=CONTENT_SCHEMA,
                system="당신은 공공기관 시설 점검 보고서를 작성하는 담당자입니다. 한국어로 작성합니다.",
            )
            content = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception as e:
            log.error("LLM 보고서 작성 실패 → 템플릿 문장 사용: %s", e)
    if content is None:
        content = template_content(metrics, cases)
    return {
        "overview": _clip(content["overview"], LIMITS["overview"]),
        "pipeline": _clip(content["pipeline"], LIMITS["pipeline"]),
        "performance": _clip(content["performance"], LIMITS["performance"]),
        "case_explanations": [_clip(s, LIMITS["case"]) for s in content["case_explanations"]][: len(cases)],
        "conclusion": _clip(content["conclusion"], LIMITS["conclusion"]),
        "action_items": [_clip(s, LIMITS["action"]) for s in content["action_items"]][:3],
    }


def template_content(metrics: dict, cases: list[dict]) -> dict:
    s = metrics["summaries"]
    a, b = s.get("detector_only", {}), s.get("detector_vlm", {})
    eff = metrics["vlm_effect"]
    conf = ", ".join(f"{CONFUSOR_KO.get(k, k)} {v.get('removed', 0)}건"
                     for k, v in metrics["fp_by_confusor"].items() if v.get("removed"))
    exps = []
    for c in cases:
        verdict = "전선" if c["vlm_is_power_line"] else CONFUSOR_KO.get(c["confusor_type"], "기타")
        ok = "정답과 일치" if c["vlm_is_power_line"] == c["truth_is_power_line"] else "정답과 불일치"
        exps.append(f"#{c['cand_id']}: {', '.join(c['evidence'][:2])} 근거로 '{verdict}'(확신도 {c['confidence']:.2f}) 판정, {ok}.")
    return {
        "overview": f"드론 영상 {metrics['n_images']}장을 대상으로 전선 검출 결과를 정답과 비교하고, "
                    f"VLM 기반 오검출 보정 효과를 확인함.",
        "pipeline": "LangSAM으로 전선 후보를 넓게 추출하고 후보별 번호를 표시한 뒤, 점수가 애매하거나 차선·그림자로 "
                    "의심되는 후보만 VLM에 전달하여 의미적 맥락(지면 위 도색, 그림자 쌍, 전주 연결 등)으로 재판정함.",
        "performance": f"F1 {a.get('f1', 0):.3f} → {b.get('f1', 0):.3f}, Precision {a.get('precision', 0):.3f} → "
                       f"{b.get('precision', 0):.3f}, Recall {a.get('recall', 0):.3f} → {b.get('recall', 0):.3f}. "
                       f"VLM이 오검출 {eff.get('fp_removed', 0)}건 제거, 정검출 {eff.get('tp_removed', 0)}건 오제거"
                       + (f" (유형별: {conf})." if conf else "."),
        "case_explanations": exps,
        "conclusion": "VLM 재판정으로 형태가 유사한 혼동 객체의 오검출이 감소하였는지 위 표로 확인함. "
                      "정검출 오제거 사례는 판정 기준 보완이 필요함.",
        "action_items": ["오제거 사례 검토 및 프롬프트 기준 보완", "VLM 선별 임계값 조정 후 재평가"],
    }
