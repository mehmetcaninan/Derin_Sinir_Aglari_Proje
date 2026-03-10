from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import ModelMeta, load_checkpoint


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


@torch.inference_mode()
def predict_characters(
    model_path: str,
    crops: list[Image.Image],
    *,
    device: str = "cpu",
) -> tuple[str, ModelMeta]:
    model, meta = load_checkpoint(model_path, device=device)
    tfm = build_transform(meta.img_size)

    if not crops:
        return "", meta

    batch = torch.stack([tfm(c) for c in crops], dim=0).to(device)
    logits = model(batch)
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

    text = "".join(meta.classes[int(i)] for i in pred)
    return text, meta


def predict_single_crop(model_path: str, crop: Image.Image, *, device: str = "cpu") -> str:
    text, _meta = predict_characters(model_path, [crop], device=device)
    return text

