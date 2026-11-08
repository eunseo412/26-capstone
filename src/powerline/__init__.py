"""드론 영상 전선 검출: 검출기 + LangSAM + VLM 검증 파이프라인."""
from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
