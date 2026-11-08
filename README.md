# 26-capstone

드론 영상 전선 검출: Object Detection + LangSAM + VLM 검증 파이프라인

- 설계 문서: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 설치

```bash
pip install -e ".[dev]"                 # 기본 (데이터, 평가, 마킹, 보고서)
pip install -e ".[claude]"              # VLM: Claude API (ANTHROPIC_API_KEY 필요)
pip install -e ".[qwen]"                # VLM: 로컬 Qwen2.5-VL (GPU 필요)
pip install -e ".[yolo]"                # Baseline: YOLO-seg
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git   # LangSAM
```

## 데이터 배치

```
dataset/frame_N/images/*.jpg           # 드론 이미지
dataset/frame_N/*.xml                  # CVAT for images 1.1 정답 (polyline)
```

## 실행 순서

```bash
# 0. 데이터 현황 + 정답 시각화 (outputs/gt_vis/)
python scripts/inspect_data.py

# 1. Baseline 학습 → Baseline 후보 저장
python scripts/train_baseline.py
python scripts/run_pipeline.py --detector yolo --no-vlm --run baseline_yolo

# 2. 제안 파이프라인 (LangSAM → 마킹 → 선별 → VLM → 융합)
python scripts/run_pipeline.py --detector langsam --vlm claude --run langsam_claude

# 3. 평가 (Baseline / LangSAM 단독 / LangSAM+VLM 비교)
python scripts/evaluate.py --run langsam_claude --baseline baseline_yolo

# 4. 1페이지 Word 보고서 → outputs/langsam_claude/report.docx
python scripts/make_report.py --run langsam_claude
```

- `--limit 3`: 앞에서 3장만 돌려 빠르게 점검
- `--vlm mock`: GPU/API 없이 흐름만 점검 (규칙 기반 가짜 VLM — 결과 수치는 의미 없음)
- `--reuse <run>`: 다른 run의 LangSAM 후보를 재사용해 VLM 단계만 다시 실행

## 테스트

```bash
pytest
```
