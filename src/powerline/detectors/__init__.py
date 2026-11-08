from .base import Detector, iter_tiles, merge_masks
from .features import add_relational_features, compute_features


def build_detector(name: str, cfg: dict) -> Detector:
    dcfg = cfg["detectors"][name]
    if name == "langsam":
        from .langsam import LangSAMDetector
        return LangSAMDetector(dcfg)
    raise ValueError(f"unknown detector: {name}")


__all__ = ["Detector", "build_detector", "iter_tiles", "merge_masks",
           "compute_features", "add_relational_features"]
