"""Claude API 백엔드 (pip install anthropic).

인증: ANTHROPIC_API_KEY 환경변수 또는 `ant auth login` 프로필.
- 출력은 JSON 스키마로 강제 (output_config.format)
- 안전 분류기 거절(stop_reason == "refusal") 시 서버측 대체 모델로 재시도하는 fallbacks 사용
"""
from __future__ import annotations

import base64
import logging

import anthropic
import cv2
import numpy as np

from .base import VLMBackend
from .prompts import SYSTEM_PROMPT, VERDICT_SCHEMA

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _to_block(img: np.ndarray) -> dict:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("image encode failed")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": base64.standard_b64encode(buf.tobytes()).decode()},
    }


class ClaudeBackend(VLMBackend):
    name = "claude"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = anthropic.Anthropic()

    def _create(self, system: str, content: list, schema: dict | None) -> str:
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=self.cfg.get("max_tokens", 16000),
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": content}],
        )
        output_config = {"effort": self.cfg.get("effort", "medium")}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs["output_config"] = output_config

        try:
            if self.cfg.get("fallbacks", True):
                resp = self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}, **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            raise  # SDK가 이미 재시도함(max_retries) — 상위에서 처리
        except anthropic.APIStatusError as e:
            log.error("Claude API error %s (request_id=%s): %s",
                      e.status_code, getattr(e, "request_id", None), e.message)
            raise

        if resp.stop_reason == "refusal":
            cat = resp.stop_details.category if resp.stop_details else None
            raise RuntimeError(f"Claude declined the request (category={cat})")
        if resp.stop_reason == "max_tokens":
            log.warning("응답이 max_tokens에서 잘림 — max_tokens를 늘리세요")
        return next((b.text for b in resp.content if b.type == "text"), "")

    def _ask(self, images, prompt):
        content = [_to_block(i) for i in images] + [{"type": "text", "text": prompt}]
        return self._create(SYSTEM_PROMPT, content, VERDICT_SCHEMA)

    def generate_text(self, prompt, schema=None, system=None):
        return self._create(system or "You write concise official reports in Korean.",
                            [{"type": "text", "text": prompt}], schema)
