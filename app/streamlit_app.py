from __future__ import annotations

import io
import os
import sys

import streamlit as st
from PIL import Image

# Ensure project root (one level above this file) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from braille_convert.braille import to_braille
from braille_convert.predict import predict_characters
from braille_convert.segment import segment_characters


st.set_page_config(page_title="OCR → Braille", layout="wide")

st.title("Görüntüden Karakter Tanıma → Braille Dönüşümü")

with st.sidebar:
    st.header("Ayarlar")
    model_path = st.text_input("Model yolu", value="models/ocr_cnn.pt")
    device = st.selectbox("Cihaz", options=["cpu", "cuda"], index=0)

uploaded = st.file_uploader("Metin görüntüsü yükleyin (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Bir görüntü yükleyin. Sonra sistem karakterleri segmente edip tahmin edecek ve Braille çıktısını gösterecek.")
    st.stop()

raw = uploaded.read()
img = Image.open(io.BytesIO(raw)).convert("RGB")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Yüklenen Görüntü")
    st.image(img, use_container_width=True)

with st.spinner("Karakterler ayrıştırılıyor..."):
    crops = segment_characters(img)

with st.spinner("Model ile tahmin ediliyor..."):
    try:
        text, _meta = predict_characters(model_path, crops, device=device)
    except FileNotFoundError:
        st.error(f"Model bulunamadı: `{model_path}`. Önce `scripts/train.py` ile modeli eğitin veya doğru yolu verin.")
        st.stop()

braille = to_braille(text)

with col2:
    st.subheader("Çıktılar")
    st.markdown("**Tanınan metin**")
    st.code(text if text else "(boş)", language=None)
    st.markdown("**Braille (Unicode)**")
    st.code(braille if braille else "(boş)", language=None)

st.divider()
st.subheader("Segmentlenen karakterler")

if not crops:
    st.warning("Karakter bulunamadı. Kontrastı artırmayı veya daha temiz arka planlı bir görüntü kullanmayı deneyin.")
else:
    cols = st.columns(10)
    for i, c in enumerate(crops):
        cols[i % 10].image(c, caption=str(i), use_container_width=True)

