"""Stage 7-2: 1페이지 공식 보고서(.docx) 렌더링. 서식은 전부 여기서 고정한다.

구성: 제목 / 문서 정보 + 결재란 / 1. 개요 / 2. 파이프라인 동작 / 3. 성능 비교 /
      4. VLM 주요 판단 사례 / 5. 결론 및 조치 사항 / 하단 안내문
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .content import CONFUSOR_KO, GROUP_KO

FONT = "맑은 고딕"
BASE_PT = 9.5
GRAY = "D9D9D9"


def _set_font(run, size=BASE_PT, bold=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        fonts.set(qn(attr), FONT)


def _para(doc_or_cell, text="", size=BASE_PT, bold=False, align=None, space_after=2, color=None):
    p = doc_or_cell.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.1
    if align:
        p.alignment = align
    if text:
        _set_font(p.add_run(text), size, bold, color)
    return p


def _shade(cell, hex_color=GRAY):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcpr.append(shd)


def _cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=BASE_PT, shade=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    _set_font(p.add_run(str(text)), size, bold)
    if shade:
        _shade(cell)


def _widths(table, cms):
    """Word는 셀 너비, LibreOffice는 tblGrid 너비를 보므로 둘 다 설정."""
    table.autofit = False
    for col, w in zip(table.columns, cms):
        col.width = Cm(w)
    for row in table.rows:
        for cell, w in zip(row.cells, cms):
            cell.width = Cm(w)


def _heading(doc, text):
    p = _para(doc, text, size=11, bold=True, space_after=1)
    p.paragraph_format.space_before = Pt(5)
    # 제목 아래 얇은 선
    ppr = p._element.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "808080")):
        bottom.set(qn(k), v)
    bdr.append(bottom)
    ppr.append(bdr)


def render_report(path: str | Path, rcfg: dict, metrics: dict, content: dict, cases: list[dict],
                  date: dt.date | None = None) -> Path:
    date = date or dt.date.today()
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(1.6)
    sec.left_margin = sec.right_margin = Cm(1.8)
    doc.styles["Normal"].font.name = FONT
    doc.styles["Normal"].font.size = Pt(BASE_PT)

    # ---- 제목
    _para(doc, rcfg["title"], size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

    # ---- 문서 정보(좌) + 결재란(우)
    outer = doc.add_table(rows=1, cols=2)
    outer.alignment = WD_TABLE_ALIGNMENT.CENTER
    info = outer.cell(0, 0).add_table(rows=3, cols=4)
    info.style = "Table Grid"
    doc_no = f"{rcfg.get('doc_number_prefix', 'PL')}-{date:%Y%m%d}-01"
    rows = [("문서번호", doc_no, "작성일", f"{date:%Y. %m. %d.}"),
            ("작성자", rcfg.get("author") or "", "검토자", rcfg.get("reviewer") or ""),
            ("기관", rcfg.get("organization", ""), "대상", f"{metrics['n_images']}장")]
    for r, vals in enumerate(rows):
        for c, v in enumerate(vals):
            _cell_text(info.cell(r, c), v, bold=c % 2 == 0, shade=c % 2 == 0, size=8.5)
    _widths(info, [1.9, 3.6, 1.9, 3.6])
    sign = outer.cell(0, 1).add_table(rows=2, cols=3)
    sign.style = "Table Grid"
    for c, v in enumerate(("담당", "검토", "승인")):
        _cell_text(sign.cell(0, c), v, bold=True, shade=True, size=8.5)
        sign.rows[1].height = Cm(1.1)
    _widths(sign, [1.5, 1.5, 1.5])
    _widths(outer, [12.6, 4.8])
    # 셀 안에 표를 넣으면 생기는 빈 첫 문단 최소화
    for cell in (outer.cell(0, 0), outer.cell(0, 1)):
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
        _set_font(cell.paragraphs[0].add_run(""), 2)

    # ---- 1. 개요
    _heading(doc, "1. 개요")
    _para(doc, "○ " + content["overview"])

    # ---- 2. 파이프라인 동작
    _heading(doc, "2. 파이프라인 동작")
    _para(doc, "입력 영상 → ① LangSAM 후보 추출 → ② 후보 번호 마킹 → ③ 오답 의심 후보 선별 → ④ VLM 재판정 → ⑤ 결정 융합",
          size=8.5, bold=True, color="1F4E79")
    _para(doc, "○ " + content["pipeline"])

    # ---- 3. 성능 비교
    _heading(doc, "3. 성능 비교")
    order = [k for k in ("detector_all", "detector_only", "detector_vlm") if k in metrics["summaries"]]
    t = doc.add_table(rows=1 + len(order), cols=5)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, h in enumerate(("구분", "Precision", "Recall", "F1", "오검출 후보")):
        _cell_text(t.cell(0, c), h, bold=True, shade=True, size=8.5)
    for r, k in enumerate(order, start=1):
        s = metrics["summaries"][k]
        best = k == "detector_vlm"
        for c, v in enumerate((GROUP_KO.get(k, k), f"{s['precision']:.3f}", f"{s['recall']:.3f}",
                               f"{s['f1']:.3f}", str(s["candidates_fp"]))):
            _cell_text(t.cell(r, c), v, bold=best, size=8.5)
    _widths(t, [4.6, 3.2, 3.2, 3.2, 3.2])
    eff = metrics.get("vlm_effect", {})
    conf = ", ".join(f"{CONFUSOR_KO.get(k, k)} {v.get('removed', 0)}"
                     for k, v in metrics.get("fp_by_confusor", {}).items() if v.get("removed"))
    _para(doc, f"※ 선 기반 허용오차 {metrics['tolerance_px']}px 기준. VLM 판정 {eff.get('vlm_correct', 0) + eff.get('vlm_wrong', 0)}건 중 "
               f"정답 일치 {eff.get('vlm_correct', 0)}건 / 오검출 제거 {eff.get('fp_removed', 0)}건"
               + (f" ({conf})" if conf else "") + f" / 정검출 오제거 {eff.get('tp_removed', 0)}건",
          size=8, space_after=1, color="404040")
    _para(doc, "○ " + content["performance"])

    # ---- 4. VLM 주요 판단 사례
    _heading(doc, "4. VLM 주요 판단 사례")
    if cases:
        ct = doc.add_table(rows=len(cases), cols=2)
        ct.style = "Table Grid"
        for r, (case, exp) in enumerate(zip(cases, content["case_explanations"] + [""] * len(cases))):
            img_cell, txt_cell = ct.cell(r, 0), ct.cell(r, 1)
            img_cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            if Path(case["marked_image"]).exists():
                img_cell.paragraphs[0].add_run().add_picture(case["marked_image"], width=Cm(8.4))
            else:
                _set_font(img_cell.paragraphs[0].add_run("(이미지 없음)"), 8)
            verdict = "전선" if case["vlm_is_power_line"] else CONFUSOR_KO.get(case["confusor_type"], "기타")
            truth = "전선" if case["truth_is_power_line"] else "전선 아님"
            txt_cell.text = ""
            p = txt_cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            _set_font(p.add_run(f"{case['image_id'].split('/')[-1]}  #{case['cand_id']}"), 8.5, True)
            _para(txt_cell, f"VLM 판정: {verdict} (확신도 {case['confidence']:.2f}) / 정답: {truth}", size=8.5, space_after=1)
            _para(txt_cell, exp, size=8.5, space_after=0)
        _widths(ct, [8.8, 8.6])
    else:
        _para(doc, "○ VLM 판정 대상 후보 없음")

    # ---- 5. 결론 및 조치 사항
    _heading(doc, "5. 결론 및 조치 사항")
    _para(doc, "○ " + content["conclusion"])
    for a in content["action_items"]:
        _para(doc, "  - " + a, space_after=1)

    _para(doc, "※ 본 보고서는 검출 파이프라인이 자동 생성하였으며, 수치는 평가 결과에서 직접 기입됨. 담당자 확인 후 활용 바람.",
          size=7.5, color="808080", align=WD_ALIGN_PARAGRAPH.RIGHT).paragraph_format.space_before = Pt(6)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


def count_pages(docx_path: str | Path) -> int | None:
    """LibreOffice가 있으면 PDF로 변환해 쪽수 확인 (1페이지 보장 검사용). 없으면 None."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(docx_path)],
                       check=True, capture_output=True, timeout=180)
        pdf = Path(tmp) / (Path(docx_path).stem + ".pdf")
        from pypdf import PdfReader
        return len(PdfReader(pdf).pages)
