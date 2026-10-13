"""Vòng huấn luyện chung cho các model PyTorch (Faster R-CNN, Mask R-CNN, CSRNet).

Tính năng: AMP, warmup + cosine/step, gradient clipping, validation theo MAE đếm,
lưu last/best, resume, early stopping và ghi history (json + csv).
YOLO được huấn luyện bởi Ultralytics, không đi qua lớp này.
"""
from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..data.annotations import ImageRecord
from ..data.augmentation import build_augmenter
from ..data.dataset import build_dataset, get_collate_fn
from ..models.torch_base import TorchCounter
from ..utils.io import write_json
from ..utils.seed import seed_everything
from .checkpoint import CheckpointManager
from .validation import validate

log = logging.getLogger("fruitcount")


class Trainer:
    def __init__(
        self,
        model: TorchCounter,
        train_records: Sequence[ImageRecord],
        val_records: Sequence[ImageRecord],
        aug_cfg: dict,
        out_dir: str | Path,
    ):
        self.model = model
        self.train_records = list(train_records)
        self.val_records = list(val_records)
        self.aug_cfg = aug_cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = model.train_cfg
        self.ckpt = CheckpointManager(self.out_dir)
        self.device = model.device
        self.use_amp = bool(self.cfg.get("amp", True)) and self.device.type == "cuda"

        seed_everything(int(self.cfg.get("seed", 42)))
        self.loader = self._build_loader()
        self.optimizer = self._build_optimizer()
        self.total_steps = max(1, int(self.cfg.get("epochs", 1)) * len(self.loader))
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: list[dict] = []
        self.start_epoch = 0
        self.best_mae = math.inf
        self.best_epoch = -1
        self.bad_evals = 0

    # ----- dựng thành phần -----
    def _build_loader(self) -> DataLoader:
        kind = self.model.dataset_kind
        crop = None
        pad = None
        if kind == "density":
            crop = int(self.cfg.get("crop_size", 512))
            if crop % 8:
                raise ValueError("crop_size của CSRNet phải là bội số của 8")
            pad = 8
        aug = build_augmenter(self.aug_cfg, train=True, crop_size=crop, pad_multiple=pad)
        ds = build_dataset(kind, self.train_records, aug, self.model.density_cfg)
        workers = int(self.cfg.get("num_workers", 2))
        return DataLoader(
            ds,
            batch_size=int(self.cfg.get("batch_size", 4)),
            shuffle=True,
            num_workers=workers,
            collate_fn=get_collate_fn(kind),
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=len(ds) > int(self.cfg.get("batch_size", 4)),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.trainable_parameters()
        lr = float(self.cfg.get("lr", 1e-3))
        wd = float(self.cfg.get("weight_decay", 0.0))
        name = str(self.cfg.get("optimizer", "adamw")).lower()
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=float(self.cfg.get("momentum", 0.9)), weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"Optimizer không hỗ trợ: {name}")

    def _build_scheduler(self):
        kind = str(self.cfg.get("scheduler", "cosine")).lower()
        warm = int(float(self.cfg.get("warmup_epochs", 0)) * len(self.loader))
        total, floor = self.total_steps, 0.01

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / max(1, warm)
            t = (step - warm) / max(1, total - warm)
            if kind == "cosine":
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
            if kind == "step":
                return 0.1 ** (int(t >= 0.7) + int(t >= 0.9))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    # ----- resume -----
    def _resume(self) -> None:
        st = self.ckpt.load_last(map_location=self.device)
        self.model.load_state_dict(st["model"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        if "scaler" in st and self.use_amp:
            self.scaler.load_state_dict(st["scaler"])
        self.start_epoch = int(st["epoch"]) + 1
        self.best_mae = float(st["best_mae"])
        self.best_epoch = int(st["best_epoch"])
        self.bad_evals = int(st.get("bad_evals", 0))
        self.history = list(st.get("history", []))
        log.info("Resume từ epoch %d (best val MAE %.3f)", self.start_epoch, self.best_mae)

    # ----- huấn luyện -----
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train_mode()
        grad_clip = float(self.cfg.get("grad_clip", 0) or 0)
        sums: dict[str, float] = {}
        n = 0
        for batch in self.loader:
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                losses = self.model.training_step(batch)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                log.warning("Bỏ qua batch có loss không hữu hạn (epoch %d)", epoch)
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                continue
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        return {f"train_{k}": v / max(1, n) for k, v in sums.items()}

    def fit(self, resume: bool = False) -> dict:
        epochs = int(self.cfg.get("epochs", 1))
        val_every = max(1, int(self.cfg.get("val_every", 1)))
        patience = int(self.cfg.get("patience", 0) or 0)
        if resume and self.ckpt.has_last():
            self._resume()

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            row.update(self._train_one_epoch(epoch))

            if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
                row.update(validate(self.model, self.val_records, batch_size=2))
                if row["val_mae"] < self.best_mae:
                    self.best_mae, self.best_epoch, self.bad_evals = row["val_mae"], epoch, 0
                    self.ckpt.save_best(
                        self.model.state_dict(),
                        {"epoch": epoch, "val_mae": self.best_mae, "cfg_name": self.model.cfg.get("name")},
                    )
                else:
                    self.bad_evals += 1
            row["epoch_time_s"] = round(time.time() - t0, 1)
            self.history.append(row)
            log.info(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

            self.ckpt.save_last(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "epoch": epoch,
                    "best_mae": self.best_mae,
                    "best_epoch": self.best_epoch,
                    "bad_evals": self.bad_evals,
                    "history": self.history,
                }
            )
            self._write_history()
            if patience and self.bad_evals >= patience:
                log.info("Early stopping tại epoch %d (không cải thiện %d lần validation)", epoch, patience)
                break

        if self.ckpt.best_path.exists():  # trả model về trạng thái tốt nhất
            self.model.load(str(self.ckpt.best_path))
        return {"best_val_mae": self.best_mae, "best_epoch": self.best_epoch, "history": self.history}

    def _write_history(self) -> None:
        write_json(self.history, self.out_dir / "history.json")
        keys: list[str] = []
        for r in self.history:
            keys += [k for k in r if k not in keys]
        with open(self.out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
