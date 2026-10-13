import json

import cv2
import numpy as np

from src.data import (
    ImageRecord,
    export_yolo_dataset,
    load_from_coco,
    load_from_instance_masks,
    load_from_yolo_txt,
    load_records,
    masks_to_boxes,
    save_records,
)
from src.data.annotations import read_instance_mask


def test_masks_to_boxes_basic():
    m = np.zeros((20, 30), np.int32)
    m[2:6, 3:9] = 1      # cao 4, rộng 6
    m[10:15, 20:25] = 7  # id không liên tục
    boxes, ids = masks_to_boxes(m)
    assert ids.tolist() == [1, 7]
    assert boxes.tolist() == [[3, 2, 9, 6], [20, 10, 25, 15]]


def test_masks_to_boxes_min_area_and_empty():
    m = np.zeros((10, 10), np.int32)
    m[0, 0] = 1  # 1 pixel
    m[4:8, 4:8] = 2
    boxes, ids = masks_to_boxes(m, min_area=4)
    assert ids.tolist() == [2]
    b0, i0 = masks_to_boxes(np.zeros((5, 5), np.int32))
    assert b0.shape == (0, 4) and i0.shape == (0,)


def test_load_instance_masks_matches_ground_truth(synthetic, records):
    _, _, counts = synthetic
    assert len(records) == len(counts)
    for r in records:
        assert r.count == counts[r.image_id]
        assert r.width == 256 and r.height == 192
        assert (r.boxes[:, 2] > r.boxes[:, 0]).all() and (r.boxes[:, 3] > r.boxes[:, 1]).all()
        assert r.boxes.min() >= 0 and r.boxes[:, [0, 2]].max() <= r.width and r.boxes[:, [1, 3]].max() <= r.height


def test_read_instance_mask_rgb_encoding(tmp_path):
    m = np.zeros((8, 8, 3), np.uint8)
    m[1:3, 1:3] = (5, 0, 0)   # mảng ghi theo thứ tự BGR: B=5 -> id 5
    m[4:6, 4:6] = (0, 1, 0)   # G=1 -> id 256
    cv2.imwrite(str(tmp_path / "m.png"), m)
    ids = np.unique(read_instance_mask(tmp_path / "m.png"))
    assert ids.tolist() == [0, 5, 256]


def test_records_json_roundtrip(records, tmp_path):
    save_records(records, tmp_path / "r.json")
    back = load_records(tmp_path / "r.json")
    assert [r.image_id for r in back] == [r.image_id for r in records]
    for a, b in zip(records, back):
        assert np.allclose(a.boxes, b.boxes, atol=0.01)
        assert a.mask_path == b.mask_path


def test_yolo_export_roundtrip(records, tmp_path):
    subset = records[:5]
    yaml_path = export_yolo_dataset({"train": subset[:3], "val": subset[3:]}, tmp_path / "y")
    assert yaml_path.exists()
    back = load_from_yolo_txt(tmp_path / "y/images/train", tmp_path / "y/labels/train")
    assert [r.count for r in back] == [r.count for r in subset[:3]]
    for a, b in zip(subset[:3], back):
        # nhãn lưu 6 chữ số thập phân nên sai lệch nhỏ hơn 0.01 pixel
        assert np.allclose(np.sort(a.boxes, axis=0), np.sort(b.boxes, axis=0), atol=0.05)


def test_load_coco(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((40, 60, 3), np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}],
        "annotations": [
            {"image_id": 1, "bbox": [5, 6, 10, 12], "iscrowd": 0},
            {"image_id": 1, "bbox": [30, 10, 8, 8], "iscrowd": 1},  # bị bỏ qua
        ],
    }
    (tmp_path / "c.json").write_text(json.dumps(coco))
    recs = load_from_coco(tmp_path / "c.json", tmp_path)
    assert len(recs) == 1 and recs[0].count == 1
    assert recs[0].boxes[0].tolist() == [5, 6, 15, 18]


def test_record_from_dict_fills_missing_ids():
    r = ImageRecord.from_dict(
        {"image_id": "x", "image_path": "x.png", "width": 10, "height": 10, "boxes": [[0, 0, 5, 5], [1, 1, 4, 4]]}
    )
    assert r.instance_ids.tolist() == [1, 2]
