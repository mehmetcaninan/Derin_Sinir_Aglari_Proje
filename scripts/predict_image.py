from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

# Ensure project root (one level above this file) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from braille_convert.braille import to_braille
from braille_convert.predict import predict_characters
from braille_convert.segment import segment_characters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--image", required=True, help="Path to input text image")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    img = Image.open(a.image)
    crops = segment_characters(img)
    text, _meta = predict_characters(a.model, crops, device=a.device)
    braille = to_braille(text)

    print("recognized_text:", text)
    print("braille:", braille)


if __name__ == "__main__":
    main()

