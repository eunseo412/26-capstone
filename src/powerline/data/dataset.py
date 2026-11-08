"""데이터셋 로딩, 정답 마스크 생성. (학습이 없으므로 split 없이 전체를 평가에 사용)"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cvat import ImageAnnotation, parse_cvat_xml

log = logging.getLogger(__name__)

_FRAME_SUFFIX = re.compile(r"_frame_\d+$")


@dataclass
class ImageRecord:
    image_id: str          # "<frame>/<파일명>" — 전체 데이터에서 고유
    frame: str             # frame_1 ...
    path: Path             # 이미지 경로 (없을 수도 있음)
    annotation: ImageAnnotation

    @property
    def video_id(self) -> str:
        """같은 영상에서 나온 프레임을 묶는 ID (예: ANAR4318_frame_000000.jpg → ANAR4318)."""
        return _FRAME_SUFFIX.sub("", Path(self.annotation.name).stem)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def load_image(self) -> np.ndarray:
        """RGB uint8."""
        img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def gt_mask(self, thickness: int) -> np.ndarray:
        return rasterize_polylines(self.annotation, thickness)


def rasterize_polylines(ann: ImageAnnotation, thickness: int) -> np.ndarray:
    mask = np.zeros((ann.height, ann.width), dtype=np.uint8)
    for pl in ann.polylines:
        pts = np.round(pl.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_8)
    return mask.astype(bool)


def _frame_dirs(root: Path, frames) -> list[Path]:
    """frames가 "auto"면 XML이 들어 있는 dataset 하위 폴더를 모두 사용."""
    if frames == "auto" or not frames:
        return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.xml")))
    return [root / f for f in frames]


def _image_index(fdir: Path, image_dir: str) -> dict[str, Path]:
    """파일명 → 경로. 지정한 image_dir를 우선 보고, 없으면 하위 폴더 전체에서 찾는다
    (예: frame_4_mini/images_mini/)."""
    index: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    preferred = fdir / image_dir
    for d in [preferred] + sorted(p for p in fdir.rglob("*") if p.is_dir() and p != preferred):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in exts:
                index.setdefault(f.name, f)
    return index


def load_dataset(cfg: dict, root_dir: str | Path = ".") -> list[ImageRecord]:
    dcfg = cfg["data"]
    root = Path(root_dir) / dcfg["root"]
    records: list[ImageRecord] = []
    for fdir in _frame_dirs(root, dcfg.get("frames")):
        frame = fdir.name
        xmls = sorted(fdir.glob(dcfg.get("annotation_glob", "*.xml")))
        if not xmls:
            log.warning("정답 XML 없음: %s", fdir)
            continue
        index = _image_index(fdir, dcfg.get("image_dir", "images"))
        for xml_path in xmls:
            for ann in parse_cvat_xml(xml_path, dcfg.get("positive_labels")):
                # XML의 name에 경로가 붙어 있는 경우가 있음 (예: "mini_data_003/images/X.jpg") → 파일명만 사용
                ann.name = Path(ann.name).name
                path = index.get(ann.name, fdir / dcfg.get("image_dir", "images") / ann.name)
                records.append(ImageRecord(f"{frame}/{ann.name}", frame, path, ann))
    missing = sum(not r.exists for r in records)
    if missing:
        log.warning("정답은 있으나 이미지 파일이 없는 항목: %d / %d", missing, len(records))
    return records
