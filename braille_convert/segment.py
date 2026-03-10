from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentOptions:
    min_area: int = 30
    padding: int = 2
    line_merge_y_ratio: float = 0.7  # relative to median char height


def _ensure_uint8_gray(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def segment_characters(image: Image.Image, options: SegmentOptions | None = None) -> list[Image.Image]:
    """
    Very simple segmentation: threshold + connected components (contours).
    Assumes reasonably clean background / contrast.
    Returns character crops sorted reading order.
    """
    if options is None:
        options = SegmentOptions()

    gray = _ensure_uint8_gray(image)

    # Otsu threshold; ensure foreground is white for contour finding
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) > 127:
        bw = 255 - bw

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < options.min_area:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return []

    heights = np.array([h for (_, _, _, h) in boxes], dtype=np.float32)
    median_h = float(np.median(heights))
    line_tol = max(5.0, options.line_merge_y_ratio * median_h)

    # Group into lines by y-center proximity
    items = []
    for (x, y, w, h) in boxes:
        cy = y + h / 2.0
        items.append((x, y, w, h, cy))
    items.sort(key=lambda t: t[4])  # by cy

    lines: list[list[tuple[int, int, int, int, float]]] = []
    for it in items:
        if not lines:
            lines.append([it])
            continue
        if abs(it[4] - np.mean([p[4] for p in lines[-1]])) <= line_tol:
            lines[-1].append(it)
        else:
            lines.append([it])

    # Sort within each line by x
    sorted_boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        for (x, y, w, h, _cy) in line:
            sorted_boxes.append((x, y, w, h))

    pad = options.padding
    H, W = gray.shape[:2]

    crops: list[Image.Image] = []
    rgb = np.array(image.convert("RGB"))
    for (x, y, w, h) in sorted_boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        crop = rgb[y0:y1, x0:x1, :]
        crops.append(Image.fromarray(crop))

    return crops

