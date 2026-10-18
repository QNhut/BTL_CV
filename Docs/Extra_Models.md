# Các model bổ sung cho pipeline FruitCount

Với ngân sách `/kaggle/working` 18 GB, xem
[Kaggle_Resources.md](Kaggle_Resources.md) về preflight, smoke check và cách chia phiên.

Các model mới dùng cùng registry, `CountResult`, split dữ liệu, evaluator, thư mục
output và bảng so sánh. Không lấy chỉ số công bố trên dataset của tác giả để điền
vào kết quả của MinneApple. Cần chạy evaluate trên cùng tập ảnh của dự án.

## 1. Model và chế độ được tích hợp

| Khóa trong MODELS | Config | Mặc định | Fine-tune trong trainer dự án |
|---|---|---|---|
| `rtdetrv2` | `configs/models/rtdetrv2.yaml` | Fine-tune R101 thành một lớp fruit trên split train của dự án | Có |
| `rf_detr` | `configs/models/rf_detr.yaml` | Fine-tune RF-DETR Large thành một lớp fruit | Có |
| `grounding_dino` | `configs/models/grounding_dino.yaml` | Fine-tune detector với prompt `apple.` | Có, loss gốc Transformers |
| `countgd` | `configs/models/countgd.yaml` | Fine-tune text-only, không SAM/crop ensemble | Có, criterion upstream; môi trường riêng |
| `va_count` | `configs/models/va_count.yaml` | Fine-tune density; exemplar nhãn train, DINO chọn lúc test | Có, biến thể MSE + count loss |

Nguồn checkpoint/API:

- [RT-DETRv2 fruit R101](https://huggingface.co/MohamedKhayat/fruit-detector-rtdetrv2-101), [API RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2).
- [RF-DETR Large](https://huggingface.co/Roboflow/rf-detr-large), [API RF-DETR](https://huggingface.co/docs/transformers/model_doc/rf_detr).
- [Grounding DINO Base](https://huggingface.co/IDEA-Research/grounding-dino-base).
- [CountGD checkpoint](https://huggingface.co/nikigoli/CountGD), [source ứng dụng của tác giả](https://huggingface.co/spaces/nikigoli/countgd/tree/main), [repository nghiên cứu](https://github.com/niki-amini-naieni/CountGD).
- [VA-Count checkpoint](https://huggingface.co/HopooLinZ/VA-Count), [repository tác giả](https://github.com/HopooLinZ/VA-Count).

## 2. Chọn model trong notebook Kaggle

Trong `kaggle_pipeline_git.ipynb`, ví dụ:

```python
MODELS = ['yolov8', 'csrnet', 'rtdetrv2', 'rf_detr', 'grounding_dino']
EPOCHS = None  # Dùng số epoch trong YAML; đặt 1 hoặc 2 để chạy thử.
```

Notebook tự cài `requirements-extra-models.txt` nếu chọn detector Transformers mới
hoặc VA-Count. Cài đặt có thể cập nhật Transformers/PyTorch; cần môi trường có
CUDA phù hợp. Đây là nhóm dependency tùy chọn, không được thêm bắt buộc vào môi
trường chỉ chạy năm model cũ.

Cell train vẫn gọi script cho mỗi config. `mode: pretrained` chỉ lưu cấu hình và
bỏ qua train; evaluate/predict sẽ dùng checkpoint nguồn đã khai báo, không yêu
cầu `runs/<name>/weights/best.pt`. `mode: finetune` dùng trainer, best/last,
early stopping, resume và biểu đồ loss/Accuracy như các model PyTorch hiện có.

Không có đường loss/Accuracy theo epoch cho các model chỉ suy luận pretrained.
Ba config Grounding DINO/CountGD/VA-Count mặc định đã bật `mode: finetune`.
Đổi về `pretrained` nếu chỉ muốn đánh giá checkpoint nguồn, đồng thời đổi tên run
và protocol để không nhầm với kết quả fine-tune.

Thí nghiệm YOLOv8 tiling chỉ chạy khi `yolov8` có trong MODELS.

## 3. Lớp mục tiêu và prompt

Mặc định dữ liệu là MinneApple, nên RT-DETRv2 lọc `target_labels: [Apple]` và
model văn bản dùng `prompt: apple.`. Không đếm tất cả 12 lớp rau quả trên ảnh
trong khi ground truth chỉ gán nhãn táo.

Nếu đổi dataset, cập nhật target labels/prompt tương ứng. Prompt hiện dùng chung
cho mọi ảnh trong một run; chưa có nhãn/prompt riêng cho từng ảnh. Không dùng box
ground truth test làm exemplar trong adapter VA-Count. Prompt, protocol và thông
tin tune ngưỡng được ghi trong JSON đánh giá.

RT-DETRv2/RF-DETR khi fine-tune dùng COCO annotations chuyển từ box `xyxy`,
đánh số lớp fruit bằng 0. Processor của model xử lý resize/normalize, rồi đầu ra
được chuyển về tọa độ ảnh gốc trước evaluator. Fine-tune thay classifier bằng
một lớp, nên phần classifier không giữ nguyên trọng số đa lớp pretrained.

RT-DETRv2 mặc định đã bật fine-tune với cấu hình:

```yaml
name: rtdetrv2_r101_partial
type: rtdetrv2
mode: finetune
protocol: finetuned
```

Giữ các mục model/train/inference còn lại. Đổi `name` để không trộn với run
pretrained. Đây là fine-tune bằng trainer chung của dự án, không tái lập toàn bộ
recipe huấn luyện chính thức của RF-DETR/RT-DETRv2.

## 4. CountGD: source và môi trường riêng

CountGD không phải model Transformers `AutoModel` có thể nạp trực tiếp từ link
với pipeline hiện tại. Adapter dùng source của ứng dụng tác giả và safetensors
checkpoint được cung cấp riêng.

Chuẩn bị source từ thư mục project:

```bash
python scripts/setup_research_models.py --model countgd
```

Lệnh tải code vào `external/countgd`, ghi commit hash vào
`source_revision.json`, không tự tải checkpoint lớn hoặc chạy trình cài package.
Notebook cũng gọi bước tải source này khi chọn CountGD.

Upstream dùng Transformers 4.x và CUDA extension `MultiScaleDeformableAttention`,
trong khi RF-DETR mới dùng Transformers 5.x. Không cài hai bộ phiên bản đè nhau
trong cùng Python. Tạo môi trường Python 3.9/3.10 riêng theo hướng dẫn upstream;
cài dependency CLI bằng `requirements-countgd.txt`, rồi build ops trong chính
môi trường đó:

```bash
pip install -r requirements-countgd.txt
cd external/countgd/models/GroundingDINO/ops
python setup.py build install
```

Build cần compiler/CUDA tương thích PyTorch. Sau đó đặt trong `countgd.yaml`:

```yaml
python_executable: /duong/dan/countgd-env/bin/python
```

Notebook gọi train/evaluate/predict của model này bằng Python được chỉ định.
Notebook hiện tự gọi `scripts/setup_countgd_env.py` khi chưa có Python CountGD:
tạo Python 3.10 bằng uv, cài requirements-countgd.txt và build ops. Cần Internet,
GPU CUDA và compiler/CUDA toolkit tương thích. Môi trường nằm trong
`/kaggle/working/.venvs/countgd` và chiếm thêm dung lượng; có thể tạo môi trường
trước rồi đặt `python_executable` để dùng lại. Không ép cài Transformers 4.x vào
môi trường RF-DETR. Nếu bước setup/build lỗi, notebook dừng trước train.
Khi chạy CLI thủ công, gọi script bằng Python môi trường CountGD tương ứng.
Adapter kiểm tra phiên bản Transformers và ops trước khi dựng model; thiếu
dependency sẽ dừng với hướng dẫn thay vì trả kết quả giả.

Text-only adapter dùng toàn ảnh và ngưỡng confidence; không có SAM normalization,
visual prompts hoặc crop ensemble như các thí nghiệm khác của tác giả. CountGD
không xuất box AP trong evaluator vì box dự đoán dùng cho định vị/đếm không được
xem là box extent trái cây tương đương detector có giám sát. Vẫn có MAE, RMSE,
Accuracy, tune ngưỡng validation và benchmark.

Fine-tune dùng bbox của split train, chuẩn hóa cxcywh theo kích thước ảnh gốc,
resize ảnh theo cùng quy tắc suy luận, prompt một lớp và không dùng visual exemplar.
Criterion từ builder upstream thực hiện matching, focal classification, L1/GIoU
và auxiliary losses theo weight_dict. Trainer dự án đảm nhiệm optimizer,
best/last, resume và validation; không phải recipe huấn luyện đầy đủ của bài báo.

## 5. VA-Count: giới hạn của bản tích hợp

Chuẩn bị source:

```bash
python scripts/setup_research_models.py --model va_count
```

Adapter tải `checkpoint_FSC.pth` từ Hub khi khởi tạo. Source có constructor timm
cũ; lớp tương thích bỏ riêng đối số `qk_scale=None` đã bị loại khỏi timm mới,
không thay đổi trọng số hoặc kiến trúc mật độ.

**Config mặc định là một biến thể VA-Count được ghi nhãn rõ**, không phải kết quả
tái lập toàn bộ pipeline bài báo. Nó:

1. Dùng Grounding DINO với prompt để đề xuất exemplar trên chính ảnh đầu vào.
2. Lọc kích thước box hợp lệ và lấy tối đa ba box có score cao nhất; lặp exemplar
   cuối nếu chỉ có một hoặc hai. Không có proposal hợp lệ thì trả count 0.
3. Resize ảnh thành 384×384; crop exemplar từ tensor đã resize về 64×64.
4. Chạy mạng VA-Count pretrained và tính count theo `abs(sum(output)) / 60`,
   không làm tròn số đếm khi đánh giá.

Chưa dùng binary CLIP classifier được huấn luyện riêng để chọn exemplar của
tác giả, chưa có negative exemplar pipeline hoặc normalization bổ sung.
Chế độ fine-tune hiện dùng protocol `finetuned_gt_exemplars_dino_inference`, tên run
`va_count_dino_top3_apple_partial`. Nếu cần tái lập nguyên bản, cần bổ sung checkpoint
binary classifier và quy trình chọn exemplar chính thức; không gộp kết quả này
với kết quả VA-Count trong bài báo mà không ghi chú.

Fine-tune đọc dataset detection để giữ bbox sau augmentation, resize về 384×384,
lấy tối đa ba crop từ bbox train làm exemplar và tạo density Gaussian tổng bằng
số quả. Target nhân 60 theo scale checkpoint upstream. Loss là density MSE cộng
`count_loss_weight * abs(sum(output)/60 - GT)`; mặc định trọng số count là 0.01.
DINO proposer không tham gia optimizer. Đây là biến thể supervised adaptation,
không có negative exemplar/CLIP loss của tác giả. Validation/test vẫn tự chọn
exemplar qua DINO, tuyệt đối không dùng bbox GT làm exemplar.

Notebook mặc định chọn ba model mới; nên chạy từng model trong phiên riêng nếu
không đủ VRAM/đĩa. Không tăng ngân sách đĩa chỉ để bỏ qua resource guard. Grounding
DINO/CountGD/VA-Count đặt batch 1 và AMP false ban đầu để giảm lỗi số học; cần
smoke test checkpoint thật trên GPU trước khi tăng batch hoặc bật AMP.
Cả năm model mới mặc định dùng `train.scope: decoder_heads`: chỉ cập nhật
decoder/head, giữ nguyên encoder/backbone/text encoder và positional embedding cố định.
VA-Count cập nhật decoder, các nhánh xử lý exemplar và density head; DINO proposer
không nằm trong optimizer. Log train ghi số tham số trainable/tổng và phần trăm.
Đặt `train.scope: full` để cập nhật toàn bộ tham số vốn có thể train; chế độ này
bỏ qua cờ `freeze_backbone` cũ nhưng vẫn giữ positional embedding cố định của tác giả.
Chế độ partial giảm gradient và optimizer state, không giảm kích thước trọng số
checkpoint toàn mạng. Đổi `name` khi đổi scope, không resume optimizer giữa hai scope.

Benchmark VA-Count bao gồm cả bước DINO chọn exemplar và mạng mật độ. Số tham số
và dung lượng tensor bao gồm hai mạng, không chỉ riêng bộ hồi quy mật độ.

## 6. Output và so sánh công bằng

Output vẫn dùng:

```text
runs/<name>/config.yaml
runs/<name>/weights/           # Khi fine-tune
runs/<name>/training/          # Khi train và có history
runs/<name>/evaluation/
runs/<name>/predictions/
runs/comparison/
```

Cột `protocol` phân biệt fine-tuned, pretrained ngoài, zero-shot theo prompt và
biến thể exemplar. Không kết luận model tốt hơn chỉ do một run dùng nhiều nhãn
huấn luyện hơn hoặc protocol khác. Chọn ngưỡng trên validation là bước hiệu chỉnh
có nhãn; nếu cần zero-shot hoàn toàn về cả ngưỡng, dùng `--tune-on none` và ghi rõ.

Tốc độ phải đo cùng phần cứng, ảnh đầu vào, phiên bản phần mềm và chế độ xử lý.
Môi trường CountGD riêng có thể khác torch, nên cần đọc metadata của benchmark.
Với checkpoint lấy từ Hub, dung lượng được lấy từ file đã cache nếu tìm được;
không tìm thấy thì để trống thay vì ước lượng bằng số tham số.

Các project upstream sử dụng package top-level `util`/`models` trùng nhau.
Pipeline chạy từng model bằng process CLI riêng; không dựng CountGD và VA-Count
chung một process demo. Adapter phát hiện xung đột và yêu cầu chạy process riêng.

## 7. Phạm vi kiểm chứng

- Có kiểm thử registry, lọc lớp, prompt, định dạng annotation, chuyển tọa độ,
  density scale và bỏ qua train cho model pretrained bằng test không tải weights.
- Có smoke test Transformers thật với RT-DETRv2/RF-DETR nhỏ khởi tạo ngẫu nhiên
  và lưu cục bộ: forward, backward và predict; không phải test checkpoint R101/Large.
- Grounding DINO có test suy luận và fine-tune forward/backward với model/tokenizer nhỏ lưu cục bộ.
- Đã đối chiếu config và processor thật trên Hub: RT-DETRv2 fruit nhận tensor
  640×640 và RF-DETR Large nhận 704×704 với processor mặc định được kiểm tra.
- RF-DETR nhỏ đã chạy hết train → evaluate → compare qua CLI, tạo checkpoint,
  biểu đồ loss/Accuracy và bảng deployment bằng dữ liệu tổng hợp.
- Có kiểm thử mạng mật độ từ source VA-Count với kiến trúc thu nhỏ và weights
  ngẫu nhiên: predict, loss/backward, lưu/nạp checkpoint. Không thay thế việc kiểm chứng checkpoint FSC trên dữ liệu thật.
- CountGD có contract test chuyển bbox, trọng số criterion và gradient bằng mạng giả;
  criterion/CUDA backend production vẫn cần smoke check thật trên Kaggle.
- Chưa kiểm chứng toàn bộ checkpoint production/GPU hoặc build CUDA CountGD trên
  Kaggle. Dependency, tải weights và VRAM thực tế cần xác nhận trong môi trường đó.

Các test tùy chọn cần dependency mới; source nghiên cứu tải về nằm trong
`external/` và được gitignore. Không commit checkpoint hoặc source upstream vào repo.
