# BÁO CÁO TOÀN DIỆN VỀ TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION REPORT)
## Dự án: FruitCount — Đếm trái cây trên cây qua ảnh, so sánh đa kiến trúc

> 💡 **HƯỚNG DẪN XEM:** Nhấn **`Ctrl + Shift + V`** (hoặc `Ctrl + K, V`) trong VS Code để xem chế độ Markdown Preview kẻ bảng đẹp mắt.

---

## 1. BẢNG SO SÁNH CÁC PHƯƠNG PHÁP TĂNG CƯỜNG DỮ LIỆU

Bảng so sánh chi tiết giữa các phương pháp tăng cường đã được cài đặt trong dự án:

| Phương pháp | Thuộc bộ dữ liệu | Cách thức biến đổi | Xử lý Bounding Box & Mask | Mục đích kỹ thuật |
|---|---|---|---|---|
| **Mosaic 4-ảnh** | `dense_occlusion` | Ghép 4 góc từ 4 ảnh khác nhau vào canvas $1024 \times 1024$ | Cắt tỉa box/mask theo 4 góc phần tư, gán lại ID toàn cục | Tăng vọt mật độ quả (lên tới 230 quả/ảnh), tạo đường cắt biên |
| **Foliage Occlusion** | `dense_occlusion` | Vẽ đa giác/ellipse màu lá cây hoặc cành nhánh đè lên quả | Thu nhỏ box về phần quả còn hở, xóa pixel mask bị che | Buộc model học nhận diện quả bị cành lá che khuất $30\% - 70\%$ |
| **Cluster Copy-Paste** | `dense_occlusion` | Cắt cụm quả dán đè xung quanh một chùm quả mục tiêu | Thêm box mới, gán ID mới duy nhất vào instance mask | Mô phỏng chùm quả dính sát nhau, thử thách thuật toán NMS |
| **Canopy Soft Shadows**| `lighting_weather`| Phủ bóng râm làm tối mềm (Gaussian Blur $81 \times 81$) | **Giữ nguyên 100%** tọa độ box và mask | Nhận diện quả nằm sâu trong bóng tối của tán lá rậm rạp |
| **Sun Flare & Glare** | `lighting_weather`| Tạo quầng sáng mặt trời rọi xiên làm lóa trắng góc ảnh | **Giữ nguyên 100%** tọa độ box và mask | Chống hiện tượng mất dấu quả khi bị ánh nắng chiếu trực diện |
| **Adaptive CLAHE** | `lighting_weather`| Cân bằng histogram thích ứng trên kênh độ sáng L (Lab) | **Giữ nguyên 100%** tọa độ box và mask | Làm nổi rõ vân vỏ quả và đường viền quả mà không đổi màu sắc |
| **Farm Weather Sim** | `lighting_weather`| Phủ lớp tán xạ sương mù, ấm hóa hoàng hôn, âm u hóa | **Giữ nguyên 100%** tọa độ box và mask | Giúp mô hình bền vững ở mọi khung giờ chụp ngoài vườn thực tế |
| **High-Res Tile Crop** | `multiscale_tiling`| Cắt ô $640 \times 640$ bao quanh các cụm quả nhỏ ở xa | Trừ offset crop $(x_0, y_0)$, clip box về $[0, 640]$ | Giữ nguyên 100% độ phân giải gốc cho quả nhỏ, tối ưu cho SAHI |
| **Micro-Fruit Sim** | `multiscale_tiling`| Downscale $35\%-70\%$ bằng INTER_AREA rồi upscale lại | Giữ nguyên tọa độ box và mask | Buộc mạng trích xuất đặc trưng hình thái học thay vì nét viền |
| **Balanced Transform**| `balanced_hybrid` | Lật ảnh HFlip + Color Jitter nhẹ + Crop tỷ lệ diện tích | Lật trục $x$, tịnh tiến tọa độ theo khung crop | Tạo tập dữ liệu cơ sở ổn định để làm mốc đối chuẩn công bằng |

---

## 2. BẢNG SO SÁNH ĐỐI ĐẦU ƯU ĐIỂM & NHƯỢC ĐIỂM GIỮA CÁC BỘ DATA

Bảng so sánh trực diện ưu điểm, nhược điểm và tính phù hợp giữa tập Gốc và 4 tập tăng cường:

| Tiêu chí so sánh | Dữ liệu Gốc (Baseline) | Tập 1: `dense_occlusion` | Tập 2: `lighting_weather` | Tập 3: `multiscale_tiling` | Tập 4: `balanced_hybrid` |
|---|---|---|---|---|---|
| **Ưu điểm lớn nhất** | Dữ liệu thật 100%, nhãn chuẩn xác, không có biến dạng nhân tạo. | **Đẩy mật độ lên cực đại (230 quả/ảnh)**, giải quyết trực tiếp câu hỏi so sánh YOLO vs CSRNet khi chồng lấn. | **Bền vững trước thời tiết thực tế**, nhận diện tốt quả trong bóng râm tán lá và dưới nắng gắt. | **Bảo tồn 100% độ nét cho quả nhỏ**, khắc phục hoàn toàn lỗi mất chi tiết khi co ảnh. | **Cực kỳ ổn định**, cân bằng hoàn hảo giữa hình học và quang học, không gây nhiễu nhãn. |
| **Nhược điểm lớn nhất** | Thiếu ảnh quả dày đặc ($>100$ quả), dễ overfit mạng lớn, thiếu đa dạng thời tiết. | Ảnh ghép Mosaic có đường chia cắt nhân tạo; tăng $60\%$ thời gian train do nhiều quả. | Tốn tài nguyên tính toán khi sinh ảnh (Gaussian Blur lớn, chuyển đổi màu Lab). | Làm hẹp trường nhìn toàn cảnh (FOV), không thấy được toàn bộ cấu trúc tán cây. | Không tạo đột phá ở các trường hợp biên khắc nghiệt (chùm quả siêu dày hay nắng chói). |
| **Nguy cơ nhiễu nhãn** | **Không có (0%)** | **Thấp - Trung bình** (nếu lá che $>80\%$ quả) | **Rất thấp (0%)** (giữ nguyên nhãn gốc) | **Thấp** (đã lọc box cắt mép $<35\%$) | **Rất thấp (~0%)** |
| **Model hưởng lợi nhất**| Mọi model (làm mốc chuẩn) | **CSRNet** (tốt nhất) & **Mask R-CNN** | **Tất cả các model** (YOLOv8, Faster R-CNN) | **YOLOv8** & **Faster R-CNN** | **Cả 4 mô hình** |
| **Ảnh hưởng đến NMS** | Mức độ trung bình | **Thử thách cực đại**: NMS dễ triệt tiêu nhầm quả dính chùm | Không ảnh hưởng đến NMS | Giúp NMS dễ lọc hơn do quả hiển thị to và rõ nét | Ổn định bình thường |
| **Đóng góp cho bài BTL**| Thiết lập đường cơ sở (Baseline) | **Điểm nhấn học thuật lớn nhất**: Phục vụ vẽ đồ thị MAE theo mật độ | Chứng minh tính ứng dụng thực tế trên Drone/Robot ngoài đồng ruộng | Minh chứng giải pháp xử lý quả nhỏ kết hợp SAHI Tiling | Tập chuẩn mực để chạy script so sánh tổng thể toàn diện |

---

## 3. BẢNG SO SÁNH SỐ LƯỢNG & ĐẶC TRƯNG MẬT ĐỘ GIỮA CÁC DATA

📁 **Vị trí thư mục dữ liệu:** `D:\CV\BTL\Data\augmented\` (hoặc `D:\CV\BTL\BTL_CV\data\augmented\`)

| Bộ dữ liệu | Số ảnh gốc | Số ảnh mới | Tổng số ảnh | Tổng số quả GT | Quả TB / ảnh | Quả Min | Quả Max | Tập Train | Tập Val | Tập Test |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Gốc** | 670 | 0 | **670** | 28,176 | 42.1 | 1 | 123 | 470 | 100 | 100 |
| **`dense_occlusion`** | 670 | 134 | **804** | **39,062** | **48.6** | 1 | **230** | 562 | 121 | 121 |
| **`lighting_weather`** | 670 | 134 | **804** | 36,233 | 45.1 | 1 | 123 | 562 | 121 | 121 |
| **`multiscale_tiling`**| 670 | 134 | **804** | 33,347 | 41.5 | 1 | 123 | 562 | 121 | 121 |
| **`balanced_hybrid`**  | 670 | 132 | **802** | 35,343 | 44.1 | 1 | 123 | 562 | 120 | 120 |

---

## 4. BẢNG DỰ ĐOÁN HIỆU NĂNG LÝ THUYẾT TRƯỚC KHI TRAIN

Dự đoán thứ hạng hiệu quả của 4 kiến trúc mô hình (Xếp hạng từ **1 = Tốt nhất** đến **4 = Kém nhất**):

| Kịch bản thử nghiệm / Bộ Data | YOLOv8 / YOLO11 | Faster R-CNN | Mask R-CNN | CSRNet (Density) | Nguyên nhân lý thuyết cốt lõi |
|---|:---:|:---:|:---:|:---:|---|
| **Chùm quả dày đặc & chồng lấn** (`dense_occlusion`) | 3 | 4 | 2 | **1 (Tốt nhất)** | **CSRNet** không dùng NMS, các hàm Gaussian tự cộng dồn khi quả đè nhau; **YOLO** bị NMS triệt tiêu nhầm box dẫn đến đếm thiếu. |
| **Quả nhỏ ở cự ly xa** (`multiscale_tiling`) | **1 (Tốt nhất)** | 2 | 3 | 4 | **YOLOv8** có FPN/PAN đa tầng trích xuất quả nhỏ rất nhạy; **CSRNet** bị hạn chế bởi bước downsample $8\times$. |
| **Nắng chói & Bóng râm sâu** (`lighting_weather`) | 2 | 3 | 3 | **1** | Đặc trưng trường mật độ ít bị phá vỡ cấu trúc biên hơn so với việc nhận diện 4 cạnh bounding box. |
| **Độ chính xác vị trí quả (mAP)** | **1** | 2 | **1 (Mask)** | N/A | CSRNet không dự đoán bounding box nên không tính được mAP. |
| **Tốc độ suy luận (FPS / Real-time)** | **1 (Nhanh nhất)** | 3 | 4 (Chậm nhất) | 2 | YOLOv8 là mạng single-stage tối ưu GPU; Mask R-CNN tốn thời gian sinh mask cho từng instance. |

---

## 5. BẢNG HƯỚNG DẪN LỰA CHỌN THEO BÀI TOÁN & LỆNH TRAIN THỬ NGHIỆM

| Mục tiêu nghiên cứu | Data đề xuất | Model tối ưu | Lệnh huấn luyện thử nghiệm |
|---|---|---|---|
| **So sánh YOLO vs CSRNet khi chồng lấn** | `dense_occlusion` | YOLOv8 & CSRNet | `python scripts/train.py --model configs/models/yolov8.yaml --dataset configs/dataset_dense_occlusion.yaml`<br>`python scripts/train.py --model configs/models/csrnet.yaml --dataset configs/dataset_dense_occlusion.yaml --epochs 50` |
| **Đánh giá ứng dụng thực địa thời gian thực** | `lighting_weather`| YOLOv8 | `python scripts/train.py --model configs/models/yolov8.yaml --dataset configs/dataset_lighting_weather.yaml` |
| **Đánh giá đếm quả nhỏ kết hợp SAHI tiling**| `multiscale_tiling`| YOLOv8 + `--tile` | `python scripts/train.py --model configs/models/yolov8.yaml --dataset configs/dataset_multiscale_tiling.yaml`<br>`python scripts/evaluate.py --model configs/models/yolov8.yaml --split test --tile --tag _tile` |
| **Thiết lập Baseline đối chuẩn công bằng** | `balanced_hybrid` | Toàn bộ 4 model | `python scripts/train.py --model configs/models/yolov8.yaml --dataset configs/dataset_balanced_hybrid.yaml`<br>`python scripts/train.py --model configs/models/csrnet.yaml --dataset configs/dataset_balanced_hybrid.yaml --epochs 50` |

---

> 🖼️ **Xem ảnh minh họa kết quả tăng cường:** Toàn bộ ảnh showcase so sánh trực quan Box | Mask | Density đã được lưu sẵn tại:
> 📁 **`D:\CV\BTL\BTL_CV\runs\augmentation_showcase`**
