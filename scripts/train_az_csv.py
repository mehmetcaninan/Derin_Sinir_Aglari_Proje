from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Ensure project root (one level above this file) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from braille_convert.model import ModelMeta, SimpleOCRCNN, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    csv_path: Path
    out_path: Path
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    device: str


class AZCsvDataset(Dataset):
    """
    Kaggle: A-Z Handwritten Alphabets in .csv format
    File: A_Z Handwritten Data.csv
    - Column 0: label in [0, 25] => 'A'..'Z'
    - Columns 1-784: pixel values (0-255) of 28x28 grayscale image, row-major.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, img_size: int):
        assert X.shape[1] == 28 * 28, "Expected 784 pixel columns (28x28)"
        self.X = X.astype(np.float32) / 255.0
        self.y = y.astype(np.int64)
        self.img_size = img_size

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        img = self.X[idx].reshape(1, 28, 28)  # (C,H,W)
        if self.img_size != 28:
            # simple nearest resize via torch
            t = torch.from_numpy(img).unsqueeze(0)  # (1,1,28,28)
            img_t = torch.nn.functional.interpolate(t, size=(self.img_size, self.img_size), mode="nearest")[
                0
            ]
            img = img_t.numpy()
        return torch.from_numpy(img), int(self.y[idx])


def load_az_csv(csv_path: Path, img_size: int, val_ratio: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    data = df.to_numpy()
    y = data[:, 0]
    X = data[:, 1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=seed,
        stratify=y,
    )

    train_ds = AZCsvDataset(X_train, y_train, img_size=img_size)
    val_ds = AZCsvDataset(X_val, y_val, img_size=img_size)
    return train_ds, val_ds


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
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
    train_ds, val_ds = load_az_csv(cfg.csv_path, img_size=cfg.img_size)

    classes = [chr(ord("A") + i) for i in range(26)]
    model = SimpleOCRCNN(num_classes=len(classes)).to(cfg.device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(cfg.device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(cfg.device != "cpu"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        print(
            f"epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                str(cfg.out_path),
                model,
                ModelMeta(img_size=cfg.img_size, classes=classes),
                extra={"best_val_acc": best_val_acc},
            )

    print(f"saved_best_model={cfg.out_path} best_val_acc={best_val_acc:.4f}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train CNN on Kaggle 'A_Z Handwritten Data.csv' (A-Z handwritten characters)."
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to 'A_Z Handwritten Data.csv' from Kaggle dataset 'A-Z Handwritten Alphabets in .csv format'",
    )
    p.add_argument("--out", default="models/ocr_cnn.pt", help="Output model path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=32, help="Network input size (will resize from 28x28)")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    a = p.parse_args()
    return TrainConfig(
        csv_path=Path(a.csv),
        out_path=Path(a.out),
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        img_size=a.img_size,
        device=a.device,
    )


if __name__ == "__main__":
    train(parse_args())

