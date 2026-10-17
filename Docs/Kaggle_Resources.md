# Kiểm tra model và ngân sách tài nguyên Kaggle

Notebook dùng **18 GB làm ngân sách do người dùng chọn**, không giả định mọi
phiên Kaggle có cùng quota. Số GB ở tài liệu này là 10⁹ byte. Ổ đĩa, RAM CPU và
VRAM GPU là ba giới hạn khác nhau; còn chỗ trên ổ đĩa không bảo đảm train không
bị CUDA out-of-memory.

## 1. Các thay đổi để giảm rủi ro

- Dataset YOLO dùng symlink tới ảnh trong `/kaggle/input`, không copy toàn bộ ảnh
  vào `/kaggle/working/processed`. Nhãn TXT/JSON vẫn được ghi trong working.
- `HF_HOME` và `TORCH_HOME` được đặt trong `/kaggle/working/.cache` để cache có
  vị trí rõ ràng và được tính vào ngân sách. Các model dùng chung cache Hub.
- Tắt cache chunk Xet bổ sung và cache pip trong notebook; không tự xóa cache
  hoặc checkpoint đã có của người dùng.
- `LOW_MEMORY_MODE = True`: train batch 1, DataLoader workers 0. Đánh giá cũng
  dùng batch 1. Tham số thực tế được lưu vào `runs/<name>/config.yaml`.
- YOLO đặt `cache=False`, `save_period=-1`; không lưu thêm checkpoint theo từng
  epoch định kỳ ngoài best/last. Artifact/biểu đồ vẫn được giữ.
- Mỗi model chạy bằng process CLI riêng và lần lượt; kết thúc process sẽ giải
  phóng bộ nhớ model, không giữ cả mười model đồng thời trong notebook.
- Evaluator mặc định chỉ giữ số đếm/box/score, bỏ mask và density map từng ảnh
  sau suy luận để không tích lũy toàn bộ các mảng ảnh trong RAM. Khi chủ động
  yêu cầu `return_results=True` hoặc `keep_visuals=True`, các mảng vẫn được giữ.

Symlink phụ thuộc việc dataset Input còn được gắn vào phiên. Khi chuyển phiên
hoặc đổi đường dẫn Input, chạy lại bước prepare; không coi symlink là bản sao dữ liệu.

## 2. Kiểm tra trước khi train

Cell train gọi `scripts/check_resources.py` sau bước chuẩn bị dependency/source.
Script kiểm tra:

1. Dung lượng hiện có trong working và dung lượng trống thật của filesystem.
2. Tổng mức dự trù của các model được chọn, cộng phần dự phòng 2 GB.
3. Import dependency/kiến trúc, môi trường Python riêng của CountGD và CUDA ops.
4. Phiên bản torch, GPU/VRAM trống và RAM khả dụng khi có psutil.

Kết quả lưu tại `/kaggle/working/runs/resource_plan.json`. Thiếu dependency hoặc
không đủ ngân sách sẽ dừng trước vòng train dài. Lệnh này không tải weights để
kiểm tra import; nó chưa chứng minh forward/backward với checkpoint production.

```bash
python scripts/check_resources.py --models yolov8 csrnet rf_detr \
  --working-root /kaggle/working --budget-gb 18 --reserve-gb 2 --check-imports
```

### Mức dự trù ổ đĩa

Đây là **allowance bảo thủ để lập kế hoạch**, không phải dung lượng weights đo
thực tế, không phải VRAM cần thiết và không phải cận trên bảo đảm cho mọi phiên bản.
Nó dự trù source cache, best/last/checkpoint tạm và artifact vừa phải:

| Loại model | Dự trù thêm (GB) |
|---|---:|
| YOLO, mỗi model | 0.8 |
| CSRNet | 1.8 |
| Faster R-CNN | 1.5 |
| Mask R-CNN | 1.7 |
| RT-DETRv2 | 3.5 |
| RF-DETR | 2.0 |
| Grounding DINO | 1.5 |
| CountGD | 3.5 |
| VA-Count + DINO | 2.0 |

Chọn cả mười model cho tổng allowance 19.1 GB, chưa gồm working hiện có và
2 GB dự phòng, nên preflight sẽ từ chối trong ngân sách 18 GB. Mức này có thể
tính dư khi cache/checkpoint đã tồn tại hoặc model chỉ chạy pretrained; chủ ý
là phát hiện sớm kế hoạch quá chật, không khẳng định mười model thực tế luôn
chiếm đúng 19.1 GB.

Dataset copy cũ, virtual environment cài trong working, output các phiên trước
và checkpoint tải thủ công vẫn được tính vào working hiện có. Allowance CountGD
không bao gồm việc tạo mới toàn bộ một môi trường PyTorch/CUDA thứ hai.

Nên chia thành nhiều phiên: nhóm năm model cũ; nhóm RT-DETRv2/RF-DETR/Grounding
DINO; CountGD và VA-Count ở các phiên đã chuẩn bị đúng môi trường. Có thể gom
`metrics_*.json` và CSV đánh giá từ các phiên về đúng thư mục model rồi chạy
`compare.py`; để so sánh tốc độ, vẫn cần điều kiện phần cứng/phần mềm tương đương.

## 3. Smoke check với model thật trong phiên Kaggle

`SMOKE_TEST_NEW_MODELS = True` mặc định chạy `scripts/check_model.py` cho từng
model mới được chọn trước vòng train dài. Script:

- Nạp checkpoint thực theo config, suy luận một ảnh validation.
- Với RT-DETRv2/RF-DETR ở `mode: finetune`, chạy thêm một backward/optimizer step
  trên batch 1 để cấp phát cả optimizer state; dùng AMP theo cấu hình khi CUDA có sẵn.
- Không ghi đè best/last; lưu báo cáo `evaluation/smoke_check.json` khi thành công.
- Chạy trong process riêng, sau đó đóng process để giải phóng RAM/VRAM.

```bash
python scripts/check_model.py --model configs/models/rf_detr.yaml \
  --out /kaggle/working/runs --backward
```

Smoke check chỉ thử một ảnh/batch. Nó giúp phát hiện lỗi tải weights, API,
dependency và thiếu bộ nhớ sớm; không bảo đảm mọi ảnh nhiều vật thể hơn hoặc
toàn bộ 50 epoch đều vừa RAM/VRAM. Nếu bước này lỗi, xem traceback trước khi
train dài; không tự bỏ qua model lỗi rồi ghi kết quả so sánh giả.

CountGD vẫn cần Python/Transformers 4.x và ops CUDA riêng. VA-Count vẫn là biến
thể DINO top-3 được mô tả trong [Extra_Models.md](Extra_Models.md), chưa tái lập
bộ lọc CLIP gốc. Các kiểm tra tài nguyên không loại bỏ những yêu cầu đó.

## 4. Kiểm tra trong khi lưu checkpoint

Các biến môi trường notebook bật cơ chế bảo vệ:

```text
FRUITCOUNT_WORKING_ROOT=/kaggle/working
FRUITCOUNT_DISK_BUDGET_GB=18
FRUITCOUNT_MIN_FREE_GB=2
```

Trainer PyTorch ước lượng kích thước tensor model/optimizer trước khi ghi file
tạm; kiểm tra chỗ cho file mới trong khi last/best cũ vẫn còn. Sau khi ghi xong
mới đổi tên nguyên tử. Nếu ghi lỗi, chỉ file tạm của lần ghi đó được dọn; checkpoint
cũ không bị xóa. YOLO kiểm tra phần dự trù checkpoint ở đầu mỗi epoch.

Khi không đặt biến môi trường, các script local giữ cách chạy bình thường.
Guard không kiểm soát được mọi file thư viện ghi giữa hai lần kiểm tra hoặc
process ngoài pipeline cùng ghi ổ đĩa; luôn cần theo dõi dung lượng thực tế.

## 5. Phạm vi đã kiểm chứng tại máy phát triển

Đã chạy test giao diện các adapter; RT-DETRv2/RF-DETR/Grounding DINO thật với
kiến trúc nhỏ và weights cục bộ; VA-Count với source upstream và mạng thu nhỏ.
RF-DETR nhỏ đã qua CLI train/evaluate/compare và smoke-check có optimizer step.
Có test quota, checkpoint không bị mất khi guard/ghi file lỗi và evaluator không
giữ toàn bộ mask/density map. Chưa có GPU Kaggle trong môi trường phát triển,
nên chưa thể xác nhận checkpoint R101/Large/CountGD production vừa VRAM của phiên bạn.
