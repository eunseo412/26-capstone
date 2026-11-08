"""스크립트 공통: 설정 로딩, 로깅, 대상 이미지 선택."""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powerline import load_config  # noqa: E402
from powerline.data import load_dataset  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--frames", nargs="*", default=None, help="일부 폴더만 (예: --frames frame_1 frame_2)")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (빠른 점검용)")
    return p


def setup(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.frames:
        cfg["data"]["frames"] = args.frames
    records = load_dataset(cfg, ROOT)
    if args.limit:
        records = records[: args.limit]
    return cfg, records
