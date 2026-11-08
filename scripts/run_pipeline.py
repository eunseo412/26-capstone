"""Stage 1~5 실행 (학습 없음, 사전학습 모델 조합만 사용).

예)
  # LangSAM 후보만 추출 (VLM 없이) → 'LangSAM 단독' 기준 방식 결과도 이것으로 평가됨
  python scripts/run_pipeline.py --no-vlm --run langsam_only
  # 추출해 둔 후보를 재사용해 VLM 판정
  python scripts/run_pipeline.py --reuse langsam_only --vlm claude --run langsam_claude
  python scripts/run_pipeline.py --reuse langsam_only --vlm qwen --run langsam_qwen
"""
import shutil

from _common import ROOT, base_parser, setup
from powerline.detectors import build_detector
from powerline.pipeline import Pipeline
from powerline.vlm import build_vlm


def main():
    p = base_parser(__doc__)
    p.add_argument("--detector", default="langsam", choices=["langsam"])
    p.add_argument("--vlm", default=None, help="qwen | claude | mock (기본: 설정 파일 = qwen)")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--run", required=True, help="outputs/<run> 이름")
    p.add_argument("--reuse", default=None, help="다른 run의 후보(candidates) 재사용")
    args = p.parse_args()
    cfg, recs = setup(args)
    run_dir = ROOT / cfg["output"]["root"] / args.run
    detector = None
    if args.reuse:
        src = ROOT / cfg["output"]["root"] / args.reuse / "candidates"
        if not src.is_dir():
            raise SystemExit(
                f"재사용할 후보가 없습니다: {src}\n"
                f"먼저 LangSAM 후보를 추출하세요:\n"
                f"  python scripts/run_pipeline.py --no-vlm --run {args.reuse}"
                + (f" --limit {args.limit}" if args.limit else ""))
        shutil.copytree(src, run_dir / "candidates", dirs_exist_ok=True)
    else:
        detector = build_detector(args.detector, cfg)
    vlm = None if args.no_vlm else build_vlm(cfg, args.vlm)
    Pipeline(cfg, detector, vlm, run_dir, use_vlm=not args.no_vlm).run(recs)


if __name__ == "__main__":
    main()
