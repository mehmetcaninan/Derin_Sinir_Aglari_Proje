from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class SimpleOCRCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass(frozen=True)
class ModelMeta:
    img_size: int
    classes: list[str]


def save_checkpoint(
    path: str,
    model: nn.Module,
    meta: ModelMeta,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "meta": {
            "img_size": meta.img_size,
            "classes": meta.classes,
        },
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[nn.Module, ModelMeta]:
    ckpt = torch.load(path, map_location=device)
    classes = list(ckpt["meta"]["classes"])
    img_size = int(ckpt["meta"]["img_size"])
    model = SimpleOCRCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ModelMeta(img_size=img_size, classes=classes)

