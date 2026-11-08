"""Baseline YOLO-seg 학습: 정답 polyline → 타일 polygon 데이터 생성 후 학습."""
from _common import ROOT, base_parser, setup
from powerline.data import split_by_video
from powerline.data.to_yolo import export_yolo_seg


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", default="yolo11s-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--export-only", action="store_true")
    args = p.parse_args()
    args.split = "all"
    cfg, recs = setup(args)
    s = cfg["data"]["split"]
    sp = split_by_video(recs, s["val_ratio"], s["test_ratio"], s["seed"])
    ycfg = cfg["detectors"]["yolo"]
    data_yaml = export_yolo_seg({"train": sp["train"], "val": sp["val"]}, ROOT / "outputs/yolo_data",
                                tile=ycfg["tile"]["size"], overlap=ycfg["tile"]["overlap"],
                                thickness=cfg["gt"]["line_thickness"])
    print("YOLO 데이터:", data_yaml)
    if args.export_only:
        return
    from ultralytics import YOLO
    YOLO(args.model).train(data=str(data_yaml), epochs=args.epochs, imgsz=ycfg["imgsz"],
                           project=str(ROOT / "runs"), name="baseline", exist_ok=True)


if __name__ == "__main__":
    main()
