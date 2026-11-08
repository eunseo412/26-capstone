"""데이터 현황 출력 + 정답 polyline 시각화 (outputs/gt_vis/)."""
import collections

import cv2

from _common import ROOT, base_parser, setup


def main():
    p = base_parser(__doc__)
    p.add_argument("--vis", type=int, default=5, help="시각화할 이미지 수")
    args = p.parse_args()
    cfg, recs = setup(args)
    print(f"이미지 {len(recs)}장 (파일 있음 {sum(r.exists for r in recs)}장), "
          f"영상 {len({r.video_id for r in recs})}개")
    print("프레임별:", dict(collections.Counter(r.frame for r in recs)))
    print("라벨별:", dict(collections.Counter(pl.label for r in recs for pl in r.annotation.polylines)))
    n = [len(r.annotation.polylines) for r in recs]
    if n:
        print(f"이미지당 polyline: 평균 {sum(n)/len(n):.1f}, 최소 {min(n)}, 최대 {max(n)}")

    out = ROOT / cfg["output"]["root"] / "gt_vis"
    out.mkdir(parents=True, exist_ok=True)
    for r in [r for r in recs if r.exists][: args.vis]:
        img = cv2.imread(str(r.path))
        m = r.gt_mask(cfg["gt"]["line_thickness"])
        img[m] = (0, 0, 255)
        cv2.imwrite(str(out / f"{r.frame}_{r.annotation.name}"), img)
    print("시각화 저장:", out)


if __name__ == "__main__":
    main()
