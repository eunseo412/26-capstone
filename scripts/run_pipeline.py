"""Stage 1~5 실행.

예)
  # 제안 파이프라인: LangSAM + Claude VLM
  python scripts/run_pipeline.py --detector langsam --vlm claude --run langsam_claude
  # Baseline 후보만 저장 (VLM 없이)
  python scripts/run_pipeline.py --detector yolo --no-vlm --run baseline_yolo
  # 후보 캐시 재사용, VLM만 바꿔 재실행
  python scripts/run_pipeline.py --reuse langsam_claude --vlm qwen --run langsam_qwen
"""
import shutil

from _common import ROOT, base_parser, setup
from powerline.detectors import build_detector
from powerline.pipeline import Pipeline
from powerline.vlm import build_vlm


def main():
    p = base_parser(__doc__)
    p.add_argument("--detector", default="langsam", choices=["langsam", "yolo"])
    p.add_argument("--vlm", default=None, help="claude | qwen | mock (기본: 설정 파일)")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--run", required=True, help="outputs/<run> 이름")
    p.add_argument("--reuse", default=None, help="다른 run의 후보(candidates) 재사용")
    args = p.parse_args()
    cfg, recs = setup(args)
    run_dir = ROOT / cfg["output"]["root"] / args.run
    detector = None
    if args.reuse:
        src = ROOT / cfg["output"]["root"] / args.reuse / "candidates"
        shutil.copytree(src, run_dir / "candidates", dirs_exist_ok=True)
    else:
        detector = build_detector(args.detector, cfg)
    vlm = None if args.no_vlm else build_vlm(cfg, args.vlm)
    Pipeline(cfg, detector, vlm, run_dir, use_vlm=not args.no_vlm).run(recs)


if __name__ == "__main__":
    main()
