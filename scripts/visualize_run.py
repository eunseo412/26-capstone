"""실행 결과 진단: 이미지별 정답/후보/판정 오버레이 + 후보별 CSV → results/<run>/

results/ 는 git에 올릴 수 있다 (outputs/ 는 제외됨). 공유하려면:
  git add results/<run> && git commit -m "results" && git push origin eunseo
"""
import json
import shutil

from _common import ROOT, base_parser, setup
from powerline.eval.visualize import visualize_run


def main():
    p = base_parser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=1920)
    args = p.parse_args()
    cfg, recs = setup(args)
    run_dir = ROOT / cfg["output"]["root"] / args.run
    if not (run_dir / "decisions").is_dir():
        raise SystemExit(f"결과가 없습니다: {run_dir}/decisions")
    out = visualize_run(run_dir, recs, cfg, ROOT / "results" / args.run, args.width)
    for f in ("metrics.json", "report.docx"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out / f)
    (out / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    n_img = len(list((out / "images").glob("*.jpg")))
    n_wrong = len(list((out / "vlm_wrong").glob("*.jpg")))
    print(f"저장: {out}  (이미지 {n_img}장, VLM 오판 사례 {n_wrong}장, candidates.csv)")


if __name__ == "__main__":
    main()
