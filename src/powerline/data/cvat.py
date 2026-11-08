"""CVAT for images 1.1 XML 파서.

데이터 예:
    <image id="0" name="ANAR4318_frame_000000.jpg" width="3840" height="2160">
      <polyline label="powerline" occluded="0" points="x1,y1;x2,y2;..." z_order="0"/>
    </image>
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Polyline:
    label: str
    points: np.ndarray        # (N, 2) float32, (x, y)
    occluded: bool = False


@dataclass
class ImageAnnotation:
    name: str
    width: int
    height: int
    polylines: list[Polyline] = field(default_factory=list)


def parse_points(text: str) -> np.ndarray:
    pts = [tuple(map(float, p.split(","))) for p in text.strip().split(";") if p]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def parse_cvat_xml(path: str | Path, positive_labels: list[str] | None = None) -> list[ImageAnnotation]:
    """XML 1개를 읽어 이미지별 정답 목록을 반환.

    positive_labels가 주어지면 해당 라벨의 polyline만 남긴다.
    polyline 외의 도형(box, polygon)이 있으면 polygon은 외곽선을 polyline으로 취급하고 box는 무시한다.
    """
    root = ET.parse(path).getroot()
    out: list[ImageAnnotation] = []
    for img in root.iter("image"):
        ann = ImageAnnotation(
            name=img.get("name"),
            width=int(img.get("width")),
            height=int(img.get("height")),
        )
        for shape in img:
            if shape.tag not in ("polyline", "polygon"):
                continue
            label = shape.get("label", "")
            if positive_labels and label not in positive_labels:
                continue
            pts = parse_points(shape.get("points", ""))
            if shape.tag == "polygon" and len(pts) > 2:
                pts = np.vstack([pts, pts[:1]])
            if len(pts) < 2:
                continue
            ann.polylines.append(Polyline(label=label, points=pts, occluded=shape.get("occluded") == "1"))
        out.append(ann)
    return out
