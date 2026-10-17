# FruitCount — Đếm trái cây trên cây qua ảnh, so sánh nhiều kiến trúc

Hệ thống đếm số quả trên cây từ ảnh, dùng chung một pipeline dữ liệu và một bộ chỉ số để so sánh công bằng
**đếm theo phát hiện (object detection)** với **đếm bằng hồi quy mật độ (density regression)**.

| Model | `type` | Họ | Cách đếm | Huấn luyện bởi |
|---|---|---|---|---|
| YOLOv8 / YOLO11 | `yolo` | detection | số box có score ≥ ngưỡng | Ultralytics |
| Faster R-CNN (R50-FPN) | `faster_rcnn` | detection | số box có score ≥ ngưỡng | `src/training` |
| Mask R-CNN (R50-FPN) | `mask_rcnn` | segmentation | số instance mask | `src/training` |
| CSRNet (VGG16 + dilated) | `csrnet` | density | tổng density map | `src/training` |

Câu hỏi nghiên cứu: *object detection có chính xác hơn density regression khi quả chồng lấn nhiều không?*
→ `scripts/evaluate.py` xuất MAE/RMSE **theo nhóm số quả mỗi ảnh**, `scripts/compare.py` vẽ đường cong so sánh.

## Cấu trúc

```
FruitCount/
├── configs/
│   ├── dataset.yaml            # đường dẫn dữ liệu, định dạng annotation, tỉ lệ chia
│   ├── augmentation.yaml       # augmentation cho Faster/Mask R-CNN và CSRNet
│   └── models/                 # mỗi model một file: yolov8, yolo11, faster_rcnn, mask_rcnn, csrnet
├── src/
│   ├── data/                   # annotations, splits, augmentation, density, dataset, yolo_export
│   ├── models/                 # adapter YOLO / Faster R-CNN / Mask R-CNN / CSRNet + registry
│   ├── training/               # Trainer, validation, checkpoint (resume)
│   ├── inference/              # CountingPredictor, hậu xử lý (NMS, tiling kiểu SAHI)
│   ├── evaluation/             # mAP, MAE, RMSE, FPS, phân tích lỗi, evaluator
│   ├── visualization/          # training curves, box / mask / density overlay, so sánh model
│   └── utils/
├── scripts/                    # prepare_data, train, evaluate, predict, compare, demo_app, run_all.sh
├── notebooks/kaggle_pipeline.ipynb
└── tests/
```

Mọi model trả về cùng kiểu `CountResult`, nên `evaluation/` và demo không cần biết model bên dưới.
Thêm model mới = viết một lớp kế thừa `BaseCounter` (hoặc `TorchCounter` nếu tự huấn luyện) và đăng ký trong
`src/models/registry.py`.

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python >= 3.9
```

Trên Kaggle/Colab, torch và torchvision thường đã có sẵn; chỉ cần `pip install ultralytics`.

## Dữ liệu

1. Tải MinneApple (hoặc bộ đếm trái cây khác) vào `data/raw/`.
2. Sửa `configs/dataset.yaml`:
   - `format: instance_masks` (MinneApple; mỗi quả một giá trị nguyên trong ảnh mask), `yolo_txt`, hoặc `coco`.
   - `paths.images_dir`, `paths.masks_dir` (và `mask_suffix` nếu tên mask khác tên ảnh).
3. Chạy:

```bash
python scripts/prepare_data.py
```

Lệnh này đọc annotation → `data/processed/records.json`, chia train/val/test **theo ảnh, stratify theo số quả**
(`splits.json`), và xuất dataset định dạng Ultralytics vào `data/processed/yolo/`.
**Mọi model dùng cùng một bộ split**, nên số liệu so sánh được với nhau.

Ghi chú về nhãn:
- Box là nguồn sự thật. Density map GT (CSRNet) được sinh từ **tâm box** sau khi augmentation, nên luôn khớp ảnh.
- Mask R-CNN cần annotation dạng instance mask. Với `yolo_txt` / `coco` (chỉ có box) hãy bỏ Mask R-CNN khỏi so sánh.
- MinneApple chỉ có táo. Nếu thêm ảnh cam/xoài Việt Nam, gán nhãn từng quả (box) rồi dùng `format: yolo_txt` hoặc `coco`.

## Huấn luyện

```bash
python scripts/train.py --model configs/models/yolov8.yaml
python scripts/train.py --model configs/models/faster_rcnn.yaml
python scripts/train.py --model configs/models/csrnet.yaml --epochs 50
python scripts/train.py --model configs/models/csrnet.yaml --resume      # tiếp tục từ weights/last.pt
```

Kết quả nằm ở `runs/<tên model>/`: `config.yaml` (cấu hình đã dùng), `weights/best.pt`, `weights/last.pt`,
`history.json|csv`, `training_curves.png`. Model PyTorch chọn `best.pt` theo **MAE đếm trên tập val**
(với model phát hiện, MAE lấy tại ngưỡng đếm tốt nhất trên val).

## Đánh giá

```bash
python scripts/evaluate.py --model configs/models/yolov8.yaml --split test --speed
python scripts/evaluate.py --model configs/models/yolov8.yaml --split test --tile --tag _tile
```

- **mAP@0.5, mAP@0.75, mAP@[0.5:0.95]** (model phát hiện), **MAE, RMSE, MAPE, bias, R², sai số tổng**, **FPS**.
- Với model phát hiện, **ngưỡng đếm được chọn trên val** rồi áp lên test (không tune trên test).
- `--tile` bật suy luận theo ô chồng nhau (kiểu SAHI) cho quả nhỏ / ảnh lớn.
- Xuất `metrics_<split>.json`, `per_image_<split>.csv`, `error_by_count_<split>.csv` (MAE theo mật độ quả),
  `pred_vs_gt_<split>.png`.

Gộp tất cả model thành bảng + biểu đồ cho báo cáo:

```bash
python scripts/compare.py --split test      # -> runs/comparison/*.csv|md|png
```

Chạy toàn bộ pipeline một lần: `bash scripts/run_all.sh [EPOCHS]`.

## Dự đoán và demo

```bash
python scripts/predict.py --model configs/models/yolov8.yaml --source anh_cay.jpg
python scripts/predict.py --model configs/models/csrnet.yaml --source thu_muc_anh/ --out outputs/csrnet
python scripts/demo_app.py --models configs/models/yolov8.yaml configs/models/csrnet.yaml
```

Ảnh minh họa có box (hoặc mask / heatmap density) và dòng "Fruits: N".

## Kaggle

Mở `notebooks/kaggle_pipeline.ipynb`: upload thư mục project thành một Kaggle Dataset, Add Data cho bộ MinneApple,
bật GPU, chạy lần lượt các cell. Kaggle chỉ giữ `/kaggle/working`, nên mọi output được ghi ở đó;
nếu phiên bị ngắt, chạy lại với `--resume`.

## Kiểm thử

```bash
pytest                     # toàn bộ
pytest tests/test_metrics.py tests/test_augmentation.py    # nhanh, không cần torch
```

Test dùng dataset giả sinh tự động (quả tròn trên nền xanh), không cần dữ liệu thật hay trọng số pretrained.
Test cần torch / ultralytics tự bỏ qua nếu chưa cài.

## Lưu ý khi báo cáo kết quả

- Mục tiêu mAP@0.5 ≥ 0,85 và MAE ≤ 3 quả/ảnh là tham vọng với ảnh nhiều quả nhỏ, bị che khuất. Nên báo cáo thêm
  RMSE, MAPE và MAE theo nhóm mật độ để đánh giá công bằng.
- mAP chỉ có cho model phát hiện; so sánh chung giữa các họ model nên dựa trên **MAE / RMSE / MAPE**.
- Density regression cần chọn `sigma` khi sinh density map (`configs/models/csrnet.yaml › density`); đây là siêu tham số
  nên ghi rõ trong phần phương pháp.
- Cố định `seed`, giữ nguyên split và lưu `runs/<model>/config.yaml` để tái lập kết quả.
# BTL_CV

Các model mở rộng (RT-DETRv2, RF-DETR, Grounding DINO, CountGD, VA-Count),
chế độ fine-tune/pretrained và cách chuẩn bị dependency được mô tả tại
[Docs/Extra_Models.md](Docs/Extra_Models.md). CountGD cần môi trường riêng;
VA-Count hiện là biến thể có ghi rõ chính sách chọn exemplar, không phải tái lập đầy đủ bài báo.


## Pipeline Kaggle lấy source từ GitHub

Mở `notebooks/kaggle_pipeline_git.ipynb`, bật Internet và GPU. Cell đầu lấy code từ
`https://github.com/QNhut/BTL_CV.git`, nhánh `TLuan`; cần commit và push thay đổi trước.
Nếu checkout đã tồn tại, notebook dùng lại code hiện có. Dùng phiên mới hoặc đổi `WORK`
để lấy code mới; notebook không tự pull đè lên cấu hình dữ liệu đang dùng.

Thêm MinneApple bằng Add Input. Cell dữ liệu tự tìm một cặp `images/` và `masks/`,
ưu tiên tập train. Nếu tìm thấy nhiều cặp hoặc cấu trúc khác, đặt rõ `IMAGES_DIR`
và `MASKS_DIR`. Test hiện được chia từ dữ liệu có nhãn, không đọc thư mục test ngoài riêng.
Các bước chuẩn bị, train, đánh giá và so sánh dừng ngay khi lệnh con lỗi.

`last.pt` được lưu sau mỗi epoch. Khi đã có checkpoint, notebook tự resume;
YOLO đã train xong sẽ dùng lại `best.pt`. Cần lưu output và khôi phục thư mục
`runs` vào `/kaggle/working/runs` khi chuyển phiên; code không tự sao lưu ra ngoài phiên.

Kiểm thử nhanh dùng dữ liệu tổng hợp, không đo độ chính xác trên MinneApple:

```bash
python -m pytest -q
```

Bộ test có train thử YOLOv8/YOLO11 nano, Faster R-CNN, Mask R-CNN và CSRNet trên CPU,
kiểm tra checkpoint/resume CSRNet và chạy lại YOLO đã hoàn tất. Trên Windows,
test symlink được bỏ qua nếu tài khoản không có quyền tạo symlink.
