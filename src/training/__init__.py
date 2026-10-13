"""Huấn luyện: Trainer, validation và checkpoint. Phần cần torch được import lười."""

__all__ = ["Trainer", "validate", "CheckpointManager", "save_checkpoint", "load_checkpoint"]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    if name == "validate":
        from .validation import validate

        return validate
    if name in {"CheckpointManager", "save_checkpoint", "load_checkpoint"}:
        from . import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
