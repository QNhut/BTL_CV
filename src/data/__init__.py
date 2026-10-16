"""Dữ liệu: annotation, split, augmentation, density map, Dataset và xuất YOLO.

Các thành phần cần torch (`dataset.py`) được import lười để phần còn lại
(annotation, split, augmentation, density) dùng được khi chưa cài torch.
"""
from .annotations import (
    ImageRecord,
    load_from_coco,
    load_from_instance_masks,
    load_from_yolo_txt,
    load_records,
    load_records_from_config,
    masks_to_boxes,
    save_records,
)
from .advanced_augmentation import (
    AdaptiveCLAHE,
    CanopyShadow,
    ClusterCopyPaste,
    FarmWeatherSim,
    FoliageOcclusion,
    HighResFruitTileCrop,
    MicroFruitDownscaleSim,
    SunFlare,
    create_mosaic4,
)
from .augmentation import Compose, Sample, build_augmenter
from .density import boxes_to_density, boxes_to_points
from .splits import load_split_records, load_splits, make_splits, save_splits, select_records
from .yolo_export import boxes_to_yolo_lines, export_yolo_dataset

__all__ = [
    "AdaptiveCLAHE",
    "CanopyShadow",
    "ClusterCopyPaste",
    "FarmWeatherSim",
    "FoliageOcclusion",
    "HighResFruitTileCrop",
    "MicroFruitDownscaleSim",
    "SunFlare",
    "create_mosaic4",
    "ImageRecord",
    "Sample",
    "Compose",
    "build_augmenter",
    "boxes_to_density",
    "boxes_to_points",
    "boxes_to_yolo_lines",
    "export_yolo_dataset",
    "load_from_coco",
    "load_from_instance_masks",
    "load_from_yolo_txt",
    "load_records",
    "load_records_from_config",
    "load_split_records",
    "load_splits",
    "make_splits",
    "masks_to_boxes",
    "save_records",
    "save_splits",
    "select_records",
]


def __getattr__(name):  # import lười các thứ cần torch
    if name in {
        "FruitDetectionDataset",
        "FruitDensityDataset",
        "build_dataset",
        "get_collate_fn",
        "detection_collate",
        "density_collate",
    }:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(name)
