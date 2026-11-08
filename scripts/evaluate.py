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
    m = evaluate_run(out / args.run, recs, cfg)
    (out / args.run / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    for k, s in m["summaries"].items():
        print(f"{k:15s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} FP후보={s['candidates_fp']}")
    print("VLM 효과:", m["vlm_effect"])
    print("혼동 유형별:", m["fp_by_confusor"])


if __name__ == "__main__":
    main()
