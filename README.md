# 26-capstone

드론 영상 전선 검출: **사전학습 모델 조합** (LangSAM + VLM) 파이프라인. 모델 학습은 하지 않음.

- 설계 문서: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 설치 (GPU 서버)

```bash
conda create -n powerline python=3.11 -y && conda activate powerline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 버전에 맞게
pip install -e ".[dev,qwen]"
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
python -c "import torch; print(torch.cuda.is_available())"                           # True 확인
```

기본 VLM/LLM은 로컬 **Qwen2.5-VL-7B** (무료, VRAM 16GB+). Claude API를 쓰려면
`pip install -e ".[claude]"` + `export ANTHROPIC_API_KEY=...` 후 `--vlm claude` / `--llm claude` (유료).

## 데이터 배치

```
dataset/<폴더>/*.xml                    # CVAT for images 1.1 정답 (polyline)
dataset/<폴더>/<아무 하위폴더>/*.jpg     # 드론 이미지 (파일명으로 자동 매칭)
```

## 실행 순서

```bash
# 0. 데이터 현황 + 정답 시각화 (outputs/gt_vis/)
python scripts/inspect_data.py

# 1. LangSAM 후보 추출 (VLM 없이) — 가장 오래 걸리는 단계, 한 번만
python scripts/run_pipeline.py --no-vlm --run langsam_only

# 2. 추출해 둔 후보로 VLM 판정 (1단계를 먼저 실행해야 함)
python scripts/run_pipeline.py --reuse langsam_only --run langsam_qwen

# 3. 평가: LangSAM 전체 후보 / LangSAM 단독(기준) / LangSAM+VLM(제안)
python scripts/evaluate.py --run langsam_qwen

# 4. 1페이지 Word 보고서 → outputs/langsam_qwen/report.docx
python scripts/make_report.py --run langsam_qwen
```

옵션
- `--limit 3`: 앞에서 3장만 (빠른 점검). 1~3단계에 같은 값을 줄 것
- `--frames frame_1 frame_2`: 일부 폴더만
- `--vlm claude`: Claude API (유료) / `--vlm mock`: 흐름 점검용 가짜 VLM (수치 무의미)
- `make_report.py --llm none`: LLM 없이 템플릿 문장으로 보고서 작성

## 테스트

```bash
pytest
```
