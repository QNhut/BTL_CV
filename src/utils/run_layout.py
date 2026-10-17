"""Shared paths for model artifacts, including legacy evaluation reports."""
from pathlib import Path
import shutil


def organize_training_outputs(run_dir):
    """Group Ultralytics artifacts while leaving checkpoint paths unchanged."""
    root = Path(run_dir)
    training = root / "training"
    for pattern in ("*.png", "*.jpg", "*.jpeg", "args.yaml", "results.csv", "history.json", "history.csv"):
        for source in root.glob(pattern):
            if source.is_file() and not source.name.startswith("pred_vs_gt_"):
                training.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(training / source.name))


def evaluation_reports(runs, split):
    """Prefer grouped reports; also accept old flat model directories."""
    root = Path(runs)
    selected = {}
    for pattern in (f"*/metrics_{split}*.json", f"*/evaluation/metrics_{split}*.json"):
        for path in sorted(root.glob(pattern)):
            model_dir = path.parent.parent if path.parent.name == "evaluation" else path.parent
            selected[(model_dir.name, path.name)] = path
    return sorted(selected.values())


def report_model_dir(report_path):
    path = Path(report_path)
    return path.parent.parent if path.parent.name == "evaluation" else path.parent
