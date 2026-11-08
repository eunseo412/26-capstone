"""스크립트 공통: 설정 로딩, 로깅, split 선택."""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powerline import load_config  # noqa: E402
from powerline.data import load_dataset, split_by_video  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (빠른 점검용)")
    return p


def setup(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    records = load_dataset(cfg, ROOT)
    if args.split != "all":
        s = cfg["data"]["split"]
        records = split_by_video(records, s["val_ratio"], s["test_ratio"], s["seed"])[args.split]
    if args.limit:
        records = records[: args.limit]
    return cfg, records
