"""Tăng cường dữ liệu nâng cao chuyên biệt cho bài toán FruitCount.

Thiết kế giải quyết 4 thách thức cốt lõi khi đếm trái cây ngoài vườn:
1. Dense & Occlusion: Chùm quả dày đặc, quả chồng lấn, lá cành che khuất.
2. Lighting & Foliage Shadows: Ánh sáng gắt, bóng râm tán lá, biến đổi màu sắc vỏ quả.
3. Multi-scale & High-Res Tiling: Quả nhỏ ở xa, giữ chi tiết độ phân giải cao cho SAHI/tiling.
4. Balanced Hybrid: Kết hợp hài hòa các phép biến đổi hình học và quang sai.

Tất cả phép biến đổi xử lý đồng bộ:
- Ảnh RGB uint8
- Bounding Boxes (xyxy)
- Instance Mask int32 (mỗi quả 1 ID riêng, 0 là nền)
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from .annotations import masks_to_boxes
from .augmentation import Sample, Transform, crop_sample


# --------------------------------------------------------------------------
# 1. CHIẾN LƯỢC 1: XỬ LÝ CHỒNG LẤN & MẬT ĐỘ CAO (DENSE & OCCLUSION)
# --------------------------------------------------------------------------

class FoliageOcclusion(Transform):
    """Mô phỏng lá cây hoặc cành nhánh ngẫu nhiên che phủ một phần các quả.
    
    Tạo các dải polygon hoặc ellipse màu xanh lá/nâu mờ đè lên vị trí các quả.
    Vẫn giữ box/mask nếu quả chưa bị che lấp quá mức (ngưỡng diện tích còn lại).
    """

    def __init__(
        self,
        p: float = 0.6,
        max_occlusions: int = 8,
        min_size: int = 25,
        max_size: int = 90,
    ):
        self.p = p
        self.max_occlusions = max_occlusions
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p or len(s.boxes) == 0:
            return s

        img = s.image.copy()
        mask = None if s.inst_mask is None else s.inst_mask.copy()
        h, w = s.height, s.width

        # Chọn số lượng vật che phủ
        num_occ = rng.integers(1, self.max_occlusions + 1)
        # Chọn ngẫu nhiên một số box mục tiêu để tạo che khuất thực tế
        target_indices = rng.choice(len(s.boxes), size=min(num_occ, len(s.boxes)), replace=False)

        foliage_colors = [
            (34, 100, 34),    # Xanh lá đậm (râm)
            (50, 130, 40),    # Xanh lá cây tự nhiên
            (80, 150, 60),    # Xanh non
            (70, 50, 30),     # Nâu cành cây
            (40, 60, 30),     # Xanh lục rêu
        ]

        for idx in target_indices:
            box = s.boxes[idx]
            bx = int(rng.uniform(box[0], box[2]))
            by = int(rng.uniform(box[1], box[3]))
            bx = np.clip(bx, 0, w - 1)
            by = np.clip(by, 0, h - 1)

            size_x = int(rng.integers(self.min_size, self.max_size))
            size_y = int(rng.integers(self.min_size, self.max_size))
            angle = float(rng.uniform(0, 180))

            # Màu lá/cành
            color_idx = rng.integers(0, len(foliage_colors))
            base_col = np.array(foliage_colors[color_idx], dtype=np.float32)
            noise = rng.uniform(-15, 15, size=3)
            leaf_color = np.clip(base_col + noise, 0, 255).astype(int).tolist()

            # Vẽ ellipse hoặc polygon che
            occ_mask = np.zeros((h, w), dtype=np.uint8)
            if rng.random() < 0.6:
                cv2.ellipse(occ_mask, (bx, by), (size_x // 2, size_y // 2), angle, 0, 360, 255, -1)
            else:
                # Tạo đa giác ngẫu nhiên hình chiếc lá
                pts = np.array([
                    [bx - size_x // 2, by],
                    [bx - size_x // 4, by - size_y // 2],
                    [bx + size_x // 4, by - size_y // 4],
                    [bx + size_x // 2, by],
                    [bx + size_x // 4, by + size_y // 3],
                    [bx - size_x // 4, by + size_y // 2],
                ], dtype=np.int32)
                cv2.fillPoly(occ_mask, [pts], 255)

            # Làm mờ mép một chút cho tự nhiên
            occ_mask_blurred = cv2.GaussianBlur(occ_mask, (5, 5), 0)
            alpha = (occ_mask_blurred.astype(np.float32) / 255.0)[..., None]

            # Áp che lên ảnh
            leaf_layer = np.full_like(img, leaf_color)
            img = (img * (1.0 - alpha) + leaf_layer * alpha).astype(np.uint8)

            # Cập nhật mask: vùng bị che hoàn toàn (alpha > 0.8) không còn là quả
            if mask is not None:
                mask[occ_mask > 200] = 0

        # Nếu có mask, cập nhật lại bounding boxes khớp với phần quả còn nhìn thấy
        boxes = s.boxes
        ids = s.ids
        if mask is not None and len(ids) > 0:
            new_boxes, new_ids = masks_to_boxes(mask, min_area=15)
            # Giữ thứ tự hoặc box hợp lệ
            if len(new_boxes) > 0:
                boxes, ids = new_boxes, new_ids
            else:
                boxes, ids = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return Sample(img, boxes, ids, mask)


class ClusterCopyPaste(Transform):
    """Sao chép các quả từ sample nguồn hoặc từ chính ảnh để dán tạo chùm quả dày đặc.
    
    Giúp mô phỏng chùm quả mật độ siêu cao, thử thách khả năng tách quả của Object Detection
    và khả năng tích phân mật độ của CSRNet.
    """

    def __init__(self, p: float = 0.5, max_paste_fruits: int = 15, blend_alpha: float = 0.95):
        self.p = p
        self.max_paste_fruits = max_paste_fruits
        self.blend_alpha = blend_alpha

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p or len(s.boxes) < 2 or s.inst_mask is None:
            return s

        img = s.image.copy()
        mask = s.inst_mask.copy()
        h, w = s.height, s.width
        max_id = int(mask.max()) if mask.max() > 0 else 0

        num_paste = int(rng.integers(2, min(self.max_paste_fruits, len(s.boxes)) + 1))
        src_indices = rng.choice(len(s.boxes), size=num_paste, replace=False)

        # Chọn một vùng cụm quả mục tiêu để dán chồng lấp
        cluster_target_idx = int(rng.choice(len(s.boxes)))
        target_box = s.boxes[cluster_target_idx]
        cluster_center_x = (target_box[0] + target_box[2]) / 2.0
        cluster_center_y = (target_box[1] + target_box[3]) / 2.0

        new_boxes_list = list(s.boxes)
        new_ids_list = list(s.ids)

        for src_i in src_indices:
            src_id = s.ids[src_i]
            fruit_mask = (mask == src_id)
            if not np.any(fruit_mask):
                continue

            sb = s.boxes[src_i]
            bx0, by0, bx1, by1 = int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3])
            fw, fh = bx1 - bx0, by1 - by0
            if fw < 10 or fh < 10:
                continue

            patch_img = img[by0:by1, bx0:bx1]
            patch_mask = fruit_mask[by0:by1, bx0:bx1]

            # Dán xung quanh tâm cụm với độ lệch nhỏ để tạo chùm quả dính nhau
            offset_x = rng.uniform(-fw * 1.2, fw * 1.2)
            offset_y = rng.uniform(-fh * 1.2, fh * 1.2)
            dst_x0 = int(round(cluster_center_x + offset_x - fw / 2.0))
            dst_y0 = int(round(cluster_center_y + offset_y - fh / 2.0))
            dst_x1 = dst_x0 + fw
            dst_y1 = dst_y0 + fh

            # Cắt nếu ra ngoài biên ảnh
            if dst_x0 < 0 or dst_y0 < 0 or dst_x1 >= w or dst_y1 >= h:
                continue

            # Áp dụng dán lên ảnh và mask
            max_id += 1
            cur_roi_img = img[dst_y0:dst_y1, dst_x0:dst_x1]
            cur_roi_mask = mask[dst_y0:dst_y1, dst_x0:dst_x1]

            # Chỉ dán pixel thuộc quả (patch_mask == True)
            paste_pixels = patch_mask
            cur_roi_img[paste_pixels] = patch_img[paste_pixels]
            cur_roi_mask[paste_pixels] = max_id

            img[dst_y0:dst_y1, dst_x0:dst_x1] = cur_roi_img
            mask[dst_y0:dst_y1, dst_x0:dst_x1] = cur_roi_mask

            new_boxes_list.append([dst_x0, dst_y0, dst_x1, dst_y1])
            new_ids_list.append(max_id)

        # Cập nhật lại danh sách box từ mask sau khi dán chồng
        final_boxes, final_ids = masks_to_boxes(mask, min_area=15)
        return Sample(img, final_boxes, final_ids, mask)


# --------------------------------------------------------------------------
# 2. CHIẾN LƯỢC 2: XỬ LÝ BIẾN ĐỔI ÁNH SÁNG & THỜI TIẾT (LIGHTING & SHADOWS)
# --------------------------------------------------------------------------

class CanopyShadow(Transform):
    """Mô phỏng bóng râm do tán lá hoặc cành cây chiếu qua bề mặt cây trái.
    
    Tạo các vùng tối mềm mại (soft shadow) dạng gradient/polygon trên ảnh,
    giúp model nhận diện tốt quả nằm trong bóng râm sâu.
    """

    def __init__(self, p: float = 0.7, min_shadow_factor: float = 0.35, max_shadow_factor: float = 0.7):
        self.p = p
        self.min_factor = min_shadow_factor
        self.max_factor = max_shadow_factor

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p:
            return s

        h, w = s.height, s.width
        # Tạo canvas bóng râm
        shadow_map = np.ones((h, w), dtype=np.float32)
        num_shadows = rng.integers(1, 4)

        for _ in range(num_shadows):
            pts_count = int(rng.integers(4, 8))
            center_x = rng.uniform(0, w)
            center_y = rng.uniform(0, h)
            radius = rng.uniform(min(h, w) * 0.2, min(h, w) * 0.6)

            angles = np.sort(rng.uniform(0, 2 * np.pi, size=pts_count))
            radii = radius * rng.uniform(0.6, 1.4, size=pts_count)
            xs = np.clip(center_x + radii * np.cos(angles), 0, w - 1).astype(np.int32)
            ys = np.clip(center_y + radii * np.sin(angles), 0, h - 1).astype(np.int32)
            poly_pts = np.stack([xs, ys], axis=1)

            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly_pts], 255)

            # Làm mờ biên mềm mại (soft shadow)
            ksize = int(rng.choice([41, 61, 81]))
            poly_blurred = cv2.GaussianBlur(poly_mask, (ksize, ksize), 0).astype(np.float32) / 255.0

            factor = float(rng.uniform(self.min_factor, self.max_factor))
            shadow_map *= (1.0 - poly_blurred * (1.0 - factor))

        shadow_map = np.clip(shadow_map, 0.25, 1.0)[..., None]
        new_img = np.clip(s.image.astype(np.float32) * shadow_map, 0, 255).astype(np.uint8)
        return Sample(new_img, s.boxes, s.ids, s.inst_mask)


class SunFlare(Transform):
    """Mô phỏng ánh sáng chói / nắng gắt rọi vào ống kính (flare & overexposure).
    
    Thách thức mô hình phân biệt quả bị lóa sáng chói trắng với nền tán cây.
    """

    def __init__(self, p: float = 0.5, max_intensity: float = 0.8):
        self.p = p
        self.max_intensity = max_intensity

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p:
            return s

        h, w = s.height, s.width
        # Đặt nguồn sáng ở gần mép ảnh hoặc góc trên
        flare_x = int(rng.uniform(0, w))
        flare_y = int(rng.uniform(0, h * 0.4))
        radius = int(rng.uniform(min(h, w) * 0.3, min(h, w) * 0.8))

        flare_layer = np.zeros((h, w, 3), dtype=np.float32)
        # Tạo quầng sáng tròn đồng tâm
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_flare = np.sqrt((x_coords - flare_x) ** 2 + (y_coords - flare_y) ** 2)

        intensity = rng.uniform(0.4, self.max_intensity)
        flare_grad = np.clip(1.0 - dist_from_flare / radius, 0, 1.0) ** 1.5 * intensity

        # Ánh sáng mặt trời hơi ngả vàng/cam nhạt
        sun_color = np.array([255, 245, 210], dtype=np.float32)
        flare_layer = flare_grad[..., None] * sun_color

        new_img = np.clip(s.image.astype(np.float32) + flare_layer, 0, 255).astype(np.uint8)
        return Sample(new_img, s.boxes, s.ids, s.inst_mask)


class AdaptiveCLAHE(Transform):
    """Cân bằng lược đồ tương phản thích ứng (CLAHE) trên không gian màu Lab.
    
    Làm rõ vân vỏ quả, gân lá và các quả nằm sâu trong bóng tối mà không làm đổi màu sắc.
    """

    def __init__(self, p: float = 0.6, clip_limit: float = 3.0, tile_grid_size: int = 8):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p:
            return s

        # Chuyển sang không gian màu Lab
        lab = cv2.cvtColor(s.image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clip = rng.uniform(1.5, self.clip_limit)
        grid = int(self.tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        out_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Sample(out_img, s.boxes, s.ids, s.inst_mask)


class FarmWeatherSim(Transform):
    """Mô phỏng điều kiện thời tiết ngoài đồng: sương mù (fog), hoàng hôn ấm, mây mù âm u."""

    def __init__(self, p: float = 0.6):
        self.p = p

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p:
            return s

        mode = rng.choice(["fog", "warm_dusk", "cloudy_cool"])
        img = s.image.astype(np.float32)

        if mode == "fog":
            # Phủ một lớp sương mù trắng nhạt
            fog_intensity = rng.uniform(0.15, 0.35)
            fog_layer = np.full_like(img, 230.0)
            img = img * (1.0 - fog_intensity) + fog_layer * fog_intensity

        elif mode == "warm_dusk":
            # Tăng sắc đỏ/vàng hoàng hôn
            warm_filter = np.array([1.15, 1.05, 0.85], dtype=np.float32)
            img = np.clip(img * warm_filter, 0, 255)

        elif mode == "cloudy_cool":
            # Trời âm u: hơi lạnh và giảm bão hòa màu
            cool_filter = np.array([0.9, 0.98, 1.1], dtype=np.float32)
            img = np.clip(img * cool_filter, 0, 255)
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 0.8  # giảm bão hòa
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        return Sample(np.clip(img, 0, 255).astype(np.uint8), s.boxes, s.ids, s.inst_mask)


# --------------------------------------------------------------------------
# 3. CHIẾN LƯỢC 3: QUẢ NHỎ & CẮT MẢNG ĐỘ PHÂN GIẢI CAO (MULTISCALE & TILING)
# --------------------------------------------------------------------------

class HighResFruitTileCrop(Transform):
    """Cắt patch độ phân giải cao tập trung vào vùng chứa quả nhỏ.
    
    Tránh hiện tượng resize ảnh lớn 1280x720 về 640x640 làm mờ quả nhỏ.
    Tối ưu cho cơ chế suy luận SAHI/tiling đã thiết kế trong dự án.
    """

    def __init__(self, tile_size: int = 640, min_fruits: int = 3, min_visibility: float = 0.4):
        self.tile_size = tile_size
        self.min_fruits = min_fruits
        self.min_visibility = min_visibility

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if len(s.boxes) == 0:
            return s

        h, w = s.height, s.width
        ts = self.tile_size

        # Nếu ảnh nhỏ hơn tile size thì không cắt
        if h <= ts and w <= ts:
            return s

        # Tìm các quả có kích thước nhỏ
        box_areas = (s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1])
        median_area = np.median(box_areas)
        small_fruit_indices = np.where(box_areas <= median_area * 1.5)[0]

        anchor_idx = int(rng.choice(small_fruit_indices if len(small_fruit_indices) > 0 else len(s.boxes)))
        target_box = s.boxes[anchor_idx]
        tx = (target_box[0] + target_box[2]) / 2.0
        ty = (target_box[1] + target_box[3]) / 2.0

        # Đặt cửa sổ tile bao quanh quả
        x0 = int(tx - rng.uniform(0.2, 0.8) * ts)
        y0 = int(ty - rng.uniform(0.2, 0.8) * ts)
        x0 = max(0, min(x0, w - ts))
        y0 = max(0, min(y0, h - ts))
        x1 = min(w, x0 + ts)
        y1 = min(h, y0 + ts)

        cropped = crop_sample(s, x0, y0, x1, y1, min_visibility=self.min_visibility)
        if len(cropped.boxes) >= self.min_fruits:
            return cropped
        return s


class MicroFruitDownscaleSim(Transform):
    """Mô phỏng quả ở khoảng cách cực xa bằng cách downscale cục bộ rồi upscale lại (Blur/Bicubic).
    
    Buộc mạng trích xuất đặc trưng hình thái học chứ không phụ thuộc vào chi tiết nét viền.
    """

    def __init__(self, p: float = 0.5, downscale_factor_range: Tuple[float, float] = (0.35, 0.7)):
        self.p = p
        self.factor_range = downscale_factor_range

    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:
        if rng.random() >= self.p:
            return s

        h, w = s.height, s.width
        factor = float(rng.uniform(self.factor_range[0], self.factor_range[1]))
        down_w = max(16, int(w * factor))
        down_h = max(16, int(h * factor))

        small_img = cv2.resize(s.image, (down_w, down_h), interpolation=cv2.INTER_AREA)
        restored_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)

        return Sample(restored_img, s.boxes, s.ids, s.inst_mask)


# --------------------------------------------------------------------------
# 4. CHIẾN LƯỢC 4: MOSAIC 4 ẢNH (CHO DENSE & MULTISCALE BENCHMARK)
# --------------------------------------------------------------------------

def create_mosaic4(
    samples: Sequence[Sample],
    target_size: Tuple[int, int] = (1024, 1024),
    rng: Optional[np.random.Generator] = None,
) -> Sample:
    """Ghép 4 mẫu thành 1 ảnh mosaic độ phân giải lớn.
    
    - Nhân 4 lượng ngữ cảnh và mật độ quả.
    - Tạo các quả bị cắt biên ở đường giao nhau 4 ảnh.
    - Cực kỳ mạnh mẽ cho cả YOLO và CSRNet học mật độ phức tạp.
    """
    if rng is None:
        rng = np.random.default_rng()

    assert len(samples) >= 4, "Cần tối thiểu 4 samples để tạo Mosaic-4"
    th, tw = target_size
    # Tâm chia cắt 4 ảnh dao động trong khoảng [0.3, 0.7] của kích thước ảnh đích
    xc = int(rng.uniform(tw * 0.35, tw * 0.65))
    yc = int(rng.uniform(th * 0.35, th * 0.65))

    out_img = np.full((th, tw, 3), 114, dtype=np.uint8)
    out_mask = np.zeros((th, tw), dtype=np.int32)
    new_boxes: List[np.ndarray] = []
    new_ids: List[int] = []

    quadrants = [
        (0, 0, xc, yc),      # Trên trái
        (xc, 0, tw, yc),     # Trên phải
        (0, yc, xc, th),     # Dưới trái
        (xc, yc, tw, th),    # Dưới phải
    ]

    global_id = 0
    for i, (qx0, qy0, qx1, qy1) in enumerate(quadrants):
        s = samples[i]
        qw = qx1 - qx0
        qh = qy1 - qy0

        # Resize ảnh mẫu vừa vặn với kích thước quadrant
        sw, sh = s.width, s.height
        scale = max(qw / sw, qh / sh)
        rw, rh = int(round(sw * scale)), int(round(sh * scale))

        r_img = cv2.resize(s.image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        r_mask = None
        if s.inst_mask is not None:
            r_mask = cv2.resize(s.inst_mask.astype(np.int32), (rw, rh), interpolation=cv2.INTER_NEAREST)

        # Cắt lấy vùng vừa đúng (qw, qh)
        crop_x0 = int(rng.integers(0, max(1, rw - qw + 1)))
        crop_y0 = int(rng.integers(0, max(1, rh - qh + 1)))

        sub_img = r_img[crop_y0:crop_y0 + qh, crop_x0:crop_x0 + qw]
        out_img[qy0:qy1, qx0:qx1] = sub_img

        # Biến đổi boxes sang tọa độ quadrant
        if len(s.boxes) > 0:
            b = s.boxes.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * scale - crop_x0 + qx0
            b[:, [1, 3]] = b[:, [1, 3]] * scale - crop_y0 + qy0

            # Giữ box nằm trong quadrant
            clip_b = b.copy()
            clip_b[:, [0, 2]] = np.clip(clip_b[:, [0, 2]], qx0, qx1)
            clip_b[:, [1, 3]] = np.clip(clip_b[:, [1, 3]], qy0, qy1)

            bw = clip_b[:, 2] - clip_b[:, 0]
            bh = clip_b[:, 3] - clip_b[:, 1]
            orig_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            keep = (bw * bh / np.maximum(orig_area, 1e-6) >= 0.3) & (bw >= 3) & (bh >= 3)

            for idx in np.nonzero(keep)[0]:
                global_id += 1
                new_boxes.append(clip_b[idx])
                new_ids.append(global_id)

                if r_mask is not None:
                    # Gán mask sang tọa độ toàn cục
                    old_id = s.ids[idx]
                    local_mask = (r_mask[crop_y0:crop_y0 + qh, crop_x0:crop_x0 + qw] == old_id)
                    out_mask[qy0:qy1, qx0:qx1][local_mask] = global_id

    final_boxes = np.asarray(new_boxes, dtype=np.float32) if len(new_boxes) else np.zeros((0, 4), dtype=np.float32)
    final_ids = np.asarray(new_ids, dtype=np.int64) if len(new_ids) else np.zeros((0,), dtype=np.int64)

    return Sample(out_img, final_boxes, final_ids, out_mask)
