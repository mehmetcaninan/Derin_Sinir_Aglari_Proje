from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from braille_convert.model import ModelMeta, SimpleOCRCNN, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    out_path: str
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    device: str
    num_workers: int


def build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _classes_from_imagefolder(ds: datasets.ImageFolder) -> list[str]:
    classes = [""] * len(ds.class_to_idx)
    for name, idx in ds.class_to_idx.items():
        classes[int(idx)] = str(name)
    return classes


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def train(cfg: TrainConfig) -> None:
    tfm = build_transforms(cfg.img_size)

    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    classes = _classes_from_imagefolder(train_ds)
    model = SimpleOCRCNN(num_classes=len(classes)).to(cfg.device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        print(
            f"epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
            save_checkpoint(
                cfg.out_path,
                model,
                ModelMeta(img_size=cfg.img_size, classes=classes),
                extra={"best_val_acc": best_val_acc},
            )

    print(f"saved_best_model={cfg.out_path} best_val_acc={best_val_acc:.4f}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset root with train/ and val/ subfolders")
    p.add_argument("--out", default="models/ocr_cnn.pt", help="Output model path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()
    return TrainConfig(
        data_dir=a.data,
        out_path=a.out,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        img_size=a.img_size,
        device=a.device,
        num_workers=a.num_workers,
    )


if __name__ == "__main__":
    train(parse_args())

