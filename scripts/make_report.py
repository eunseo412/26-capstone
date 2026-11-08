"""Stage 7: metrics.json → 1페이지 Word 보고서 (outputs/<run>/report.docx)"""
import json

from _common import ROOT, base_parser
from powerline import load_config
from powerline.report import count_pages, render_report, select_cases, write_content
from powerline.vlm import build_vlm


def main():
    p = base_parser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--llm", default=None, help="claude | qwen | none (기본: 설정 파일)")
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / cfg["output"]["root"] / args.run
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rcfg = cfg["report"]
    backend = args.llm or rcfg["llm_backend"]
    llm = None if backend == "none" else build_vlm(cfg, backend)
    cases = select_cases(metrics, rcfg["max_cases"])
    content = write_content(metrics, cases, llm)
    (run_dir / "report_content.json").write_text(json.dumps(content, ensure_ascii=False, indent=1))
    path = render_report(run_dir / "report.docx", rcfg, metrics, content, cases)
    pages = count_pages(path)
    print("보고서:", path, f"({pages}쪽)" if pages else "")
    if pages and pages > 1:
        print("경고: 1페이지를 넘음 — max_cases를 줄이거나 content 글자 수 상한을 낮추세요")


if __name__ == "__main__":
    main()
