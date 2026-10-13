#!/usr/bin/env bash
# Chạy toàn bộ pipeline so sánh: chuẩn bị dữ liệu -> train -> evaluate -> compare.
# Dùng: bash scripts/run_all.sh [EPOCHS]   (EPOCHS tùy chọn, ghi đè số epoch mọi model)
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS_ARG=()
if [[ $# -ge 1 ]]; then EPOCHS_ARG=(--epochs "$1"); fi

MODELS=(yolov8 yolo11 faster_rcnn mask_rcnn csrnet)

python scripts/prepare_data.py

for m in "${MODELS[@]}"; do
  echo "=== TRAIN $m ==="
  python scripts/train.py --model "configs/models/$m.yaml" "${EPOCHS_ARG[@]}"
  echo "=== EVAL $m ==="
  python scripts/evaluate.py --model "configs/models/$m.yaml" --split test --speed
done

# Model phát hiện + tiling (SAHI-style), lưu riêng để so sánh
python scripts/evaluate.py --model configs/models/yolov8.yaml --split test --tile --tag _tile

python scripts/compare.py --split test
