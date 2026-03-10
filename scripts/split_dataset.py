from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    val_ratio: float,
    seed: int,
    copy: bool,
) -> None:
    rng = random.Random(seed)
    classes = [d for d in input_dir.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders found in: {input_dir}")

    for cls_dir in classes:
        images = [p for p in cls_dir.iterdir() if p.is_file() and is_image(p)]
        if not images:
            continue
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
        val_imgs = set(images[:n_val])
        train_imgs = [p for p in images if p not in val_imgs]

        for split, items in [("train", train_imgs), ("val", list(val_imgs))]:
            dst_cls = output_dir / split / cls_dir.name
            dst_cls.mkdir(parents=True, exist_ok=True)
            for src in items:
                dst = dst_cls / src.name
                if dst.exists():
                    continue
                if copy:
                    shutil.copy2(src, dst)
                else:
                    os.link(src, dst)  # hardlink (fast, no extra disk if same FS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Root folder: class subfolders inside (e.g. A/, B/, ...)")
    p.add_argument("--output", required=True, help="Output folder to create train/ and val/")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy files instead of hardlink (safer across filesystems)")
    a = p.parse_args()

    split_dataset(
        Path(a.input),
        Path(a.output),
        val_ratio=a.val_ratio,
        seed=a.seed,
        copy=a.copy,
    )
    print(f"done. output={a.output}")


if __name__ == "__main__":
    main()

