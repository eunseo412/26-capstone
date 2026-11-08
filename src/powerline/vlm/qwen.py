"""로컬 Qwen2.5-VL 백엔드 (GPU 필요, 7B 기준 VRAM ~16GB+).

pip install "transformers>=4.49" accelerate qwen-vl-utils
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .base import VLMBackend
from .prompts import SYSTEM_PROMPT, VERDICT_SCHEMA


class QwenVLBackend(VLMBackend):
    name = "qwen"

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model"], torch_dtype=torch.bfloat16, device_map=cfg.get("device", "auto"))
        self.processor = AutoProcessor.from_pretrained(cfg["model"])

    def _run(self, messages: list, images: list) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.cfg.get("max_new_tokens", 1024))
        gen = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    def _ask(self, images: list[np.ndarray], prompt: str) -> str:
        pils = [Image.fromarray(i) for i in images]
        schema_hint = "Respond ONLY with a JSON object matching this schema:\n" + json.dumps(VERDICT_SCHEMA)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"} for _ in pils]
             + [{"type": "text", "text": prompt + "\n" + schema_hint}]},
        ]
        return self._run(messages, pils)

    def generate_text(self, prompt, schema=None, system=None):
        hint = ("\nRespond ONLY with JSON matching:\n" + json.dumps(schema, ensure_ascii=False)) if schema else ""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system or "You write concise official reports in Korean."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt + hint}]},
        ]
        return self._run(messages, [])
