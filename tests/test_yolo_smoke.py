"""Smoke test adapter YOLO: huấn luyện 1 epoch trên CPU với mạng yolov8n khởi tạo ngẫu nhiên
(không tải trọng số) rồi dự đoán. Bị bỏ qua nếu chưa cài ultralytics."""
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torch")

from src.data import export_yolo_dataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.base import CountResult  # noqa: E402
from src.utils import load_image  # noqa: E402

CFG = {
    "name": "yolo_test", "type": "yolo", "family": "detection",
    "model": {"weights": "yolov8n.yaml", "imgsz": 64, "device": "cpu"},
    "inference": {"min_score": 0.001, "count_threshold": 0.25, "nms_iou": 0.5, "max_det": 100,
                  "tile": {"enabled": False, "size": 128, "overlap": 0.2}},
    "train_args": {"epochs": 1, "batch": 4, "workers": 0, "plots": False, "amp": False, "seed": 0},
}


@pytest.fixture(scope="module", params=["yolov8n.yaml", "yolo11n.yaml"])
def trained(records, tmp_path_factory, request):
    root = tmp_path_factory.mktemp("yolo")
    data_yaml = export_yolo_dataset({"train": records[:8], "val": records[8:12]}, root / "data")
    cfg = {**CFG, "model": {**CFG["model"], "weights": request.param}}
    model = build_model(cfg)
    best = model.fit(data_yaml, root / "run", val_records=records[8:12])
    return model, best, root


def test_fit_writes_weights(trained):
    _, best, root = trained
    assert (root / "run" / "weights" / "last.pt").exists()
    assert best.parent == root / "run" / "weights"
    assert (root / 'run/training/loss_curves.png').exists()
    assert (root / 'run/training/accuracy_curves.png').exists()
    from src.utils import read_json
    history = read_json(root / 'run/training/history.json')
    assert 0 <= history[0]['val_acc_exact'] <= 1


def test_predict_returns_count_result(trained, records):
    model, _, _ = trained
    imgs = [load_image(r.image_path) for r in records[:2]]
    res = model.predict_images(imgs)
    assert len(res) == 2 and all(isinstance(r, CountResult) for r in res)
    assert res[0].boxes.shape[1] == 4 and len(res[0].boxes) == len(res[0].scores)
    assert model.num_parameters() > 1e6


def test_tiled_prediction_runs(trained, records):
    model, _, _ = trained
    model.inference_cfg["tile"] = {"enabled": True, "size": 128, "overlap": 0.25}
    res = model.predict_image(load_image(records[0].image_path))
    assert res.boxes.shape[1] == 4
    model.inference_cfg["tile"]["enabled"] = False


def test_load_weights_roundtrip(trained, records):
    model, _, root = trained
    other = build_model(CFG)
    other.load(str(root / "run" / "weights" / "last.pt"))
    img = load_image(records[0].image_path)
    a, b = model.predict_image(img), other.predict_image(img)
    assert len(a.boxes) == len(b.boxes)


def test_resume_completed_run_reuses_best_weights(trained):
    model, best, root = trained
    assert model.fit(root / 'data/data.yaml', root / 'run', resume=True) == best
