"""Stage 6: 평가 → outputs/<run>/metrics.json"""
import json

from _common import ROOT, base_parser, setup
from powerline.eval.analysis import evaluate_run


def main():
    p = base_parser(__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    cfg, recs = setup(args)
    out = ROOT / cfg["output"]["root"]
    run_dir = out / args.run
    if not (run_dir / "decisions").is_dir():
        lim = f" --limit {args.limit}" if args.limit else ""
        raise SystemExit(
            f"평가할 결과가 없습니다: {run_dir}/decisions\n"
            f"아래 순서로 먼저 실행하세요 (앞 단계가 성공했는지 로그 확인):\n"
            f"  1) python scripts/run_pipeline.py --no-vlm --run langsam_only{lim}\n"
            f"  2) python scripts/run_pipeline.py --reuse langsam_only --run {args.run}{lim}")
    m = evaluate_run(run_dir, recs, cfg)
    if m["n_images"] == 0:
        raise SystemExit(f"{run_dir}에 대상 이미지의 결과가 없습니다. run_pipeline과 같은 --limit/--frames를 주었는지 확인하세요.")
    (run_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    for k, s in m["summaries"].items():
        print(f"{k:15s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} FP후보={s['candidates_fp']}")
    print("VLM 효과:", m["vlm_effect"])
    print("혼동 유형별:", m["fp_by_confusor"])


if __name__ == "__main__":
    main()
