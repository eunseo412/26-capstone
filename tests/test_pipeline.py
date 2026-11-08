"""합성 데이터로 Stage 0~7 전체 흐름 점검 (GPU/API 불필요: 가짜 검출기 + mock VLM)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from powerline import load_config
from powerline.data import load_dataset, parse_cvat_xml, split_by_video
from powerline.detectors.base import Detector, iter_tiles
from powerline.eval.analysis import evaluate_run
from powerline.eval.metrics import Evaluator
from powerline.pipeline import Pipeline
from powerline.report import count_pages, render_report, select_cases, write_content
from powerline.types import Candidate, mask_to_rle, rle_to_mask
from powerline.vlm import parse_verdict
from powerline.vlm.mock import MockBackend

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
WIRES = [[(0, 100), (960, 160)], [(0, 130), (960, 190)]]   # 전선 2가닥 (어두운 선)
LANE = [(100, 540), (500, 300)]                             # 차선 (흰 굵은 선, 오검출 유도)

XML = """<?xml version='1.0' encoding='utf-8'?>
<annotations>
  <image id="0" name="{name}" width="{w}" height="{h}">
{lines}
  </image>
</annotations>"""


def _make_dataset(root: Path, n=3):
    fdir = root / "dataset" / "frame_1"
    (fdir / "images").mkdir(parents=True)
    xmls = []
    for i in range(n):
        name = f"VID{i}_frame_000000.jpg"
        img = np.full((H, W, 3), 110, np.uint8)
        for a, b in WIRES:
            cv2.line(img, a, b, (30, 30, 30), 3)
        cv2.line(img, *LANE, (250, 250, 250), 9)
        cv2.imwrite(str(fdir / "images" / name), img)
        lines = "\n".join(
            f'    <polyline label="powerline" occluded="0" points="{a[0]},{a[1]};{b[0]},{b[1]}" z_order="0"/>'
            for a, b in WIRES)
        xmls.append(XML.format(name=name, w=W, h=H, lines=lines).replace("</annotations>", ""))
    body = "".join(x.split("<annotations>")[1] for x in xmls)
    (fdir / "annotations.xml").write_text("<?xml version='1.0'?>\n<annotations>" + body + "</annotations>")


class FakeDetector(Detector):
    """전선 2개(높은 점수, 애매한 점수) + 확신 높은 차선 오검출 1개를 내는 가짜 검출기."""
    name = "fake"

    def _predict_raw(self, image):
        out = []
        for (a, b), s in zip(WIRES + [LANE], [0.8, 0.45, 0.65]):
            m = np.zeros(image.shape[:2], np.uint8)
            cv2.line(m, a, b, 1, 9 if (a, b) == tuple(LANE) else 4)
            out.append((m.astype(bool), s, "power line"))
        return out


@pytest.fixture
def env(tmp_path):
    _make_dataset(tmp_path)
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["data"]["frames"] = ["frame_1"]
    cfg["data"]["annotation_glob"] = "*.xml"
    return tmp_path, cfg


def test_rle_roundtrip():
    m = np.random.default_rng(0).random((37, 53)) > 0.7
    assert (rle_to_mask(mask_to_rle(m)) == m).all()
    m[0, 0] = True
    assert (rle_to_mask(mask_to_rle(m)) == m).all()


def test_parse_real_xml():
    xml = ROOT / "dataset/frame_1/merged_annotations.xml"
    if not xml.exists():
        pytest.skip("실데이터 XML 없음")
    anns = parse_cvat_xml(xml, ["powerline", "polyline"])
    assert len(anns) == 66 and anns[0].width == 3840
    assert sum(len(a.polylines) for a in anns) == 1633


def test_split_has_no_video_leak(env):
    root, cfg = env
    recs = load_dataset(cfg, root)
    sp = split_by_video(recs + recs, 0.3, 0.3)
    vids = [{r.video_id for r in v} for v in sp.values()]
    assert not (vids[0] & vids[1]) and not (vids[0] & vids[2]) and not (vids[1] & vids[2])


def test_tiles_cover_image():
    cov = np.zeros((2160, 3840), bool)
    for x0, y0, x1, y1 in iter_tiles(2160, 3840, 1280, 0.2):
        cov[y0:y1, x0:x1] = True
    assert cov.all()


def test_parse_verdict_tolerates_chatter():
    v = parse_verdict('설명입니다 ```json\n{"cand_id":1,"is_power_line":false,"confusor_type":"shadow",'
                      '"confidence":1.4,"evidence":["a"]}\n```', 1, "x")
    assert not v.is_power_line and v.confusor_type == "shadow" and v.confidence == 1.0
    bad = parse_verdict("no json", 2, "x")
    assert bad.confidence == 0.0


def test_end_to_end(env):
    root, cfg = env
    recs = load_dataset(cfg, root)
    assert len(recs) == 3 and all(r.exists for r in recs)

    run = root / "outputs" / "run"
    Pipeline(cfg, FakeDetector(), MockBackend(), run).run(recs)

    # 차선 후보는 정답과 매칭되지 않아야 함
    ev = Evaluator(cfg["eval"]["tolerance_px"], cfg["eval"]["instance_match_ratio"])
    from powerline.pipeline import load_candidates, safe_name
    cands = load_candidates(run / "candidates" / f"{safe_name(recs[0].image_id)}.json")
    e = ev.evaluate_image(recs[0], cands)
    assert e.cand_tp == [0, 1] and e.cand_fp == [2]

    m = evaluate_run(run, recs, cfg)
    s = m["summaries"]
    # 차선(확신 높은 오검출) 제거 → precision 상승, 점수 낮던 전선 회복 → recall 상승
    assert s["detector_vlm"]["precision"] > s["detector_only"]["precision"]
    assert s["detector_vlm"]["recall"] > s["detector_only"]["recall"]
    assert m["vlm_effect"].get("fp_removed", 0) == 3
    assert m["vlm_effect"].get("tp_recovered", 0) == 3
    assert m["vlm_effect"].get("tp_removed", 0) == 0
    assert m["fp_by_confusor"]["lane_marking"]["removed"] == 3

    cases = select_cases(m, 2)
    content = write_content(m, cases, llm=None)
    path = render_report(run / "report.docx", cfg["report"], m, content, cases)
    assert path.exists()
    pages = count_pages(path)
    if pages is not None:
        assert pages == 1
