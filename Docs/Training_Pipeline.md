# Huấn luyện model bằng pipeline Kaggle

Tài liệu mô tả code hiện tại trong [kaggle_pipeline_git.ipynb](../notebooks/kaggle_pipeline_git.ipynb). Các giá trị bên dưới là cấu hình trong repository; khi thay đổi notebook hoặc YAML, cấu hình thực tế có thể khác.

## 1. Luồng chạy

```text
Clone source từ GitHub → Cài thư viện → Kiểm tra PyTorch/CUDA
→ Xác định thư mục ảnh và mask → Chuẩn bị dữ liệu và chia split
→ Huấn luyện lần lượt model được chọn
→ Đánh giá trên test → Đánh giá YOLOv8 với tiling
→ Tổng hợp bảng/biểu đồ → Dự đoán ảnh mẫu
```

Notebook gọi script bằng `run_python()`, dùng cùng Python với kernel và `subprocess.run(..., check=True)`. Nếu một script lỗi, cell dừng; Run All không tiếp tục thành công sang các bước sau.

Hiện notebook train xong tất cả model trong danh sách rồi mới đánh giá, không đánh giá ngay sau từng model.

## 2. Source và môi trường

| Tham số | Giá trị hiện tại |
|---|---|
| Repository | `https://github.com/QNhut/BTL_CV.git` |
| Nhánh | `TLuan` |
| Source trên Kaggle | `/kaggle/working/FruitCountGit` |
| Thư viện | Cài từ `requirements.txt` |
| Dữ liệu đã xử lý | `/kaggle/working/processed` |
| Kết quả train | `/kaggle/working/runs` |

Cần bật Internet và chọn GPU trong notebook Kaggle; code kiểm tra `torch.cuda.is_available()` để hiển thị khả năng dùng CUDA. Nếu CUDA không khả dụng, model PyTorch tự chọn CPU khi không chỉ định device.

Chỉ code đã commit và push mới được clone từ GitHub. Nếu thư mục source đã tồn tại và đúng nhánh, notebook dùng lại checkout đó, không tự pull. Muốn lấy code mới, dùng phiên mới hoặc đổi `WORK` sang thư mục chưa tồn tại.

## 3. Dữ liệu đầu vào và tập test

Thêm MinneApple bằng Add Input. Notebook hiện được thiết kế cho cặp ảnh và instance mask: nền có giá trị 0, mỗi quả có một ID riêng.

```python
IMAGES_DIR = None
MASKS_DIR = None
```

Khi chưa đặt cả hai đường dẫn, notebook tìm các thư mục `images/` có thư mục `masks/` cùng cấp trong `/kaggle/input`, loại các đường dẫn có thành phần `test` hoặc `val`, ưu tiên đường dẫn có thành phần `train`. Nếu không tìm được đúng một cặp, cần đặt rõ cả `IMAGES_DIR` và `MASKS_DIR`.

Notebook ghi đường dẫn vào [dataset.yaml](../configs/dataset.yaml), sau đó gọi [prepare_data.py](../scripts/prepare_data.py):

1. Đọc instance mask, suy ra bounding box của mỗi quả.
2. Lọc instance có diện tích nhỏ hơn `20 pixel` theo cấu hình.
3. Chia theo ảnh: train 70%, validation 15%, test 15%, seed `42`; phân tầng tương đối theo số quả.
4. Lưu `records.json`, `splits.json` và xuất dữ liệu YOLO.

```text
/kaggle/working/processed/
├── records.json
├── splits.json
└── yolo/
    ├── data.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

**Test hiện được chia từ nguồn ảnh/mask đã chọn; pipeline chưa đọc thư mục test riêng của dataset.** Tất cả model dùng cùng split. Ảnh không tìm thấy mask bị bỏ qua khi đọc instance masks, vì vậy cần kiểm tra số ảnh sau prepare.

YOLO đọc dataset đã xuất; Faster R-CNN đọc ảnh và box; Mask R-CNN cần thêm mask; CSRNet sinh density map từ tâm box sau augmentation.

## 4. Chọn model và số epoch

Cell huấn luyện hiện đặt:

```python
MODELS = ['yolov8']
EPOCHS = 40
```

Như vậy mặc định chỉ train YOLOv8. Để chọn cả năm model:

```python
MODELS = ['yolov8', 'csrnet', 'yolo11', 'faster_rcnn', 'mask_rcnn']
```

`EPOCHS = 40` ghi đè số epoch trong YAML cho lần train mới. Đặt `EPOCHS = None` để dùng giá trị YAML. Số epoch là giới hạn tối đa, model có thể dừng sớm. Khi resume YOLO, trạng thái và các tham số được khôi phục theo cơ chế của Ultralytics; không xem việc đổi `EPOCHS` là bảo đảm kéo dài một run đã hoàn tất.

| Khóa trong MODELS | Config | Tên thư mục kết quả | Epoch YAML | Batch |
|---|---|---|---:|---:|
| `yolov8` | `configs/models/yolov8.yaml` | `yolov8s` | 100 | 8 |
| `yolo11` | `configs/models/yolo11.yaml` | `yolo11s` | 100 | 8 |
| `faster_rcnn` | `configs/models/faster_rcnn.yaml` | `faster_rcnn_r50_fpn_v2` | 40 | 4 |
| `mask_rcnn` | `configs/models/mask_rcnn.yaml` | `mask_rcnn_r50_fpn_v2` | 40 | 2 |
| `csrnet` | `configs/models/csrnet.yaml` | `csrnet` | 100 | 4 |

Tên thư mục lấy từ trường `name`, không nhất thiết trùng tên file config. Script train hỗ trợ `--batch-size`; cell hiện chỉ ghi đè epoch, nên batch lấy từ YAML.

## 5. Augmentation

### Faster R-CNN và Mask R-CNN

Theo [augmentation.yaml](../configs/augmentation.yaml):

| Phép biến đổi | Cấu hình |
|---|---|
| Random scale crop | Xác suất 0.7; diện tích crop 0.5–1.0 ảnh gốc; ratio 0.75–1.33 |
| Lọc box sau crop | Giữ khi còn ít nhất 35% diện tích và kích thước hợp lệ |
| Lật ngang | Xác suất 0.5 |
| Lật dọc | Tắt, xác suất 0.0 |
| Color jitter | Xác suất 0.8; brightness/contrast/saturation 0.25; hue 10 theo thang OpenCV |
| Gaussian blur | Xác suất 0.1; kernel 3 hoặc 5 |
| Resize | Thu nhỏ cạnh dài tối đa 1024, không tự phóng lớn |

Các biến đổi hình học được áp dụng đồng bộ cho ảnh, box và mask. Model R-CNN còn có bước resize nội bộ của torchvision theo `min_size=800`, `max_size=1333`.

### CSRNet

Resize cạnh dài tối đa 1024 trước, sau đó random crop `512×512`, lật ngang, color jitter, blur và pad theo bội số 8. Crop cố định thay cho random scale crop của R-CNN.

Density ground truth được sinh từ box đã biến đổi: downsample 8, `sigma_scale=0.25`, `min_sigma=1.0`. Mỗi Gaussian được chuẩn hóa tổng bằng 1, kể cả khi bị cắt ở biên.

### YOLOv8 và YOLO11

Augmentation do Ultralytics xử lý từ `train_args`, không dùng `augmentation.yaml`:

| Tham số | Giá trị |
|---|---:|
| `mosaic` | 1.0 |
| `close_mosaic` | 10 epoch cuối |
| `fliplr` / `flipud` | 0.5 / 0.0 |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.5 / 0.3 |
| `scale` / `translate` | 0.5 / 0.1 |
| `imgsz` | 1024 |

Config dự án không bật MixUp riêng. Những tùy chọn không khai báo phụ thuộc mặc định của phiên bản Ultralytics được cài.

Augmentation ngẫu nhiên ở trên dùng trong train. Đánh giá đọc ảnh gốc và áp dụng tiền xử lý suy luận của từng model, không dùng các phép biến đổi ngẫu nhiên của train.

## 6. Optimizer, loss và kỹ thuật hỗ trợ train

| Cấu hình | Faster R-CNN | Mask R-CNN | CSRNet |
|---|---|---|---|
| Pretrained | Detector/backbone COCO | Detector/backbone COCO | Frontend VGG16 ImageNet |
| Optimizer | SGD | SGD | AdamW |
| Learning rate | 0.01 | 0.01 | 0.00005 |
| Momentum | 0.9 | 0.9 | Mặc định AdamW |
| Weight decay | 0.0005 | 0.0005 | 0.0001 |
| LR scheduler | Cosine | Cosine | Cosine |
| Warmup | 1 epoch | 1 epoch | 2 epoch |
| Gradient clipping | 10.0 | 10.0 | 5.0 |
| AMP | Bật khi CUDA khả dụng | Bật khi CUDA khả dụng | Bật khi CUDA khả dụng |
| Validation | Mỗi epoch | Mỗi epoch | Mỗi epoch |
| Seed | 42 | 42 | 42 |

R-CNN tối ưu tổng loss mà detector torchvision trả về; Mask R-CNN có thêm loss mask. CSRNet dùng:

```text
loss = MSE(pred_density, GT_density × 100)
       + 0.1 × L1(sum(ReLU(pred_density)) / 100, GT_count)
```

YOLO dùng trọng số `yolov8s.pt` hoặc `yolo11s.pt`, `optimizer=auto`, `lr0=0.01`, `cos_lr=true`, seed 42 và workers 2. Loss và các kỹ thuật còn lại do Ultralytics quản lý. Optimizer tự chọn có thể điều chỉnh learning rate; `lr0` không bảo đảm là LR cuối cùng được dùng.

**Dropout:** code dự án không thêm dropout cho các model. CSRNet chỉ lấy phần convolution của VGG16, không dùng classifier chứa dropout. AMP, warmup và gradient clipping hỗ trợ tốc độ/ổn định; augmentation, weight decay và early stopping hỗ trợ hạn chế overfitting.

## 7. Early stopping và chọn best model

| Model | Patience | Tiêu chí |
|---|---:|---|
| Faster R-CNN | 10 | Validation MAE không giảm |
| Mask R-CNN | 10 | Validation MAE không giảm |
| CSRNet | 20 | Validation MAE không giảm |
| YOLOv8 / YOLO11 | 30 | Validation fitness do Ultralytics quản lý |

Với R-CNN, validation chọn ngưỡng score cho MAE đếm tốt nhất rồi dùng MAE đó so sánh epoch. CSRNet dùng tổng density map. Trainer PyTorch lưu `best.pt` khi MAE giảm và nạp lại best khi kết thúc.

YOLO không dùng MAE đếm của trainer dự án để chọn best. Vì vậy tiêu chí chọn checkpoint giữa YOLO và ba model còn lại khác nhau.

## 8. Checkpoint và resume

```text
/kaggle/working/runs/<name>/weights/
├── last.pt
└── best.pt
```

Notebook kiểm tra `last.pt` theo tên model: có file thì thêm `--resume`, chưa có thì train mới. Checkpoint được lưu sau mỗi epoch, không phải sau mỗi batch. Nếu phiên ngắt giữa epoch, phần epoch chưa được lưu cần chạy lại.

Trainer PyTorch lưu model, optimizer, scheduler, AMP scaler, epoch, lịch sử và trạng thái early stopping trong `last.pt`. Resume bắt đầu từ epoch kế tiếp. YOLO dùng cơ chế checkpoint của Ultralytics; code dự án dùng lại `best.pt` khi nhận diện checkpoint đã hoàn tất thay vì gọi resume gây lỗi.

**Lưu checkpoint trên ổ làm việc không đồng nghĩa đã sao lưu qua phiên.** Cần lưu output phiên trước, thêm output đó vào phiên mới và khôi phục đúng cấu trúc `runs` về `/kaggle/working/runs` trước cell train. Notebook chưa tự tải hoặc sao lưu checkpoint ra dịch vụ ngoài.

## 9. Đánh giá và đầu ra

Output mới được chia theo chức năng. Checkpoint giữ đường dẫn cũ để resume:

```text
/kaggle/working/runs/
├── yolov8s/                         # Tương tự cho các model khác
│   ├── config.yaml
│   ├── weights/                    # best.pt, last.pt
│   ├── training/                   # history/results, args, curves, confusion matrix, ảnh batch
│   ├── evaluation/                 # metrics, per_image, error_by_count, pred_vs_gt
│   └── predictions/                # predictions.json và ảnh *_count.jpg
└── comparison/                     # Bảng CSV/Markdown, biểu đồ so sánh các model
```

Ultralytics ghi artifact vào thư mục run trong lúc train; script gom các file
vào `training/` khi lời gọi train kết thúc hoặc phát sinh exception Python.
Nếu phiên bị ngắt cưỡng bức, có thể còn file ở thư mục gốc. Để gom output cũ
hoặc file còn sót trên Kaggle, chạy:

```python
run_python('scripts/organize_runs.py --runs /kaggle/working/runs')
```

Lệnh giữ nguyên `weights/` và `config.yaml`. Với file đánh giá đã tồn tại ở cả
hai nơi, lệnh giữ cả hai và thông báo bỏ qua; phần so sánh ưu tiên bản trong
`evaluation/` để không tạo hàng trùng.

Notebook đánh giá từng model trong `MODELS` bằng `evaluate.py --split test --speed`:

- Counting: MAE, RMSE, MAPE, bias, R², sai số tổng và độ chính xác trong các mức sai số.
- Detection/segmentation: chỉ số AP dựa trên bounding box ở các ngưỡng IoU; chưa tính mask AP cho Mask R-CNN.
- Speed: FPS, latency và số tham số.
- Error analysis: sai số từng ảnh, nhóm số quả và các trường hợp sai nhiều nhất.

Với model detection, ngưỡng đếm mặc định được chọn trên validation rồi áp dụng cho test. CSRNet không dùng score threshold. Phép tính AP trong dự án là all-point interpolation; không đồng nhất hoàn toàn với evaluator COCO chuẩn.

Cell đánh giá còn gọi YOLOv8 với tiling `640×640`, overlap 0.2 và gộp box bằng NMS, xuất hậu tố `_tile`. **Lệnh này luôn chạy kể cả khi bỏ `yolov8` khỏi MODELS**, nên cần có checkpoint YOLOv8 nếu giữ nguyên cell.

| Vị trí | Nội dung |
|---|---|
| `runs/<name>/config.yaml` | Cấu hình model dùng khi train |
| `runs/<name>/weights/` | Best và last checkpoint |
| `runs/<name>/training/history.json`, `training/history.csv`, `training/training_curves.png` | Lịch sử/biểu đồ trainer PyTorch |
| `runs/<name>/training/results.csv` và các output Ultralytics | Lịch sử YOLO |
| `runs/<name>/evaluation/metrics_test.json` | Chỉ số tổng hợp |
| `runs/<name>/evaluation/per_image_test.csv` | Sai số từng ảnh |
| `runs/<name>/evaluation/error_by_count_test.csv` | Sai số theo nhóm số quả |
| `runs/<name>/evaluation/pred_vs_gt_test.png` | Dự đoán so với nhãn |
| `runs/comparison/` | Bảng CSV/Markdown và biểu đồ so sánh |
| `/kaggle/working/runs/<name>/predictions/` | Ảnh dự đoán mẫu và JSON |

Bảng `comparison_test.csv` và `comparison_test.md` có đầy đủ MAE, RMSE, MAPE,
bias, sai số tổng tương đối, R², Accuracy exact/±1/±3, mAP50/mAP75/mAP50–95,
Precision/Recall tại best F1, best F1 và ngưỡng tương ứng, FPS, latency mean/P95,
số tham số và số ảnh. Accuracy, Precision, Recall, F1 và mAP dùng thang 0–1;
MAPE dùng phần trăm. Ô thiếu để trống trong CSV và hiện `—` trong bảng notebook/Markdown.
Các hàng sắp theo MAE tăng dần; biến thể tiling có hàng riêng.

Phần minh họa dùng tối đa 4 ảnh test, lấy mẫu với seed 0. CLI predict mặc định dùng ngưỡng trong config, chưa tự nạp ngưỡng đã tune từ báo cáo evaluate; số đếm minh họa có thể khác kết quả đánh giá.

## 10. Kiểm tra trước khi chạy dài

1. Push source lên nhánh `TLuan`, dùng checkout đúng bản cập nhật.
2. Xác nhận CUDA khả dụng và đường dẫn ảnh/mask đúng.
3. Kiểm tra số ảnh từng split sau prepare; không dùng kết quả dữ liệu giả làm kết quả nghiên cứu.
4. Chọn MODELS, epoch và batch phù hợp bộ nhớ GPU; nếu giữ cell tiling thì cần YOLOv8.
5. Chạy thử ít epoch để kiểm tra pipeline trước, rồi đặt cấu hình train đầy đủ.
6. Lưu output/checkpoint khi kết thúc phiên để có thể resume.

Các kiểm thử trong repository đã chạy trên dữ liệu tổng hợp, gồm train thử năm kiến trúc trên CPU và kiểm tra checkpoint. Chúng xác nhận chức năng cơ bản; chưa chứng minh độ chính xác hoặc khả năng chạy đủ epoch trên MinneApple thật với mọi GPU Kaggle.
