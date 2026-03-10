# Görüntü Tabanlı Karakter Tanıma ve Braille Dönüştürme

Bu repo; bir görüntüden karakterleri **tek tek** tespit edip (basit segmentasyon), her karakteri bir **CNN sınıflandırıcı** ile tanıyıp, ortaya çıkan metni **Unicode Braille** karakterlerine dönüştüren bir prototip içerir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Veri seti (Kaggle + klasör yapısı seçenekleri)

### Seçenek 1 – Kaggle CSV veri seti (önerilen, A–Z gerçek veri)

Kullanacağımız gerçek veri seti:

- **Kaggle**: `A-Z Handwritten Alphabets in .csv format`
  - Sayfa: Kaggle’da arama çubuğuna **“A-Z Handwritten Alphabets in .csv format”** yazın.
  - Dosya adı: `A_Z Handwritten Data.csv`

İndirdikten sonra bu dosyayı proje altında şu konuma koyun:

```text
data/A_Z Handwritten Data.csv
```

Sonra modeli bu csv ile eğitin:

```bash
python3 scripts/train_az_csv.py --csv "data/A_Z Handwritten Data.csv" --epochs 10 --img-size 32 --batch-size 256
```

Bu komut gerçek veriye dayalı, uçtan uca çalışan bir **CNN harf sınıflandırıcısı** üretir ve modeli `models/ocr_cnn.pt` olarak kaydeder. Arayüz ve `predict_image.py` bu modeli doğrudan kullanır.

### Seçenek 2 – Klasör bazlı kendi veri setiniz

Eğitim, `torchvision.datasets.ImageFolder` ile okunur. Veri setinizi aşağıdaki gibi yerleştirin:

```
data/dataset/
  train/
    A/  (A sınıfına ait resimler)
    B/
    ...
  val/
    A/
    B/
    ...
```

- Sınıf isimleri klasör adıdır (örn. `A`, `b`, `0`).
- Model gri-tonlama + yeniden boyutlandırma ile eğitilir.

Ham veri setiniz sadece `A/ B/ ...` şeklindeyse, eğitim/validasyon bölmek için:

```bash
python3 scripts/split_dataset.py --input path/to/raw_classes --output data/dataset --val-ratio 0.2
```

## Model eğitimi

```bash
python3 scripts/train.py --data data/dataset --epochs 10 --img-size 32 --batch-size 128
```

Eğitim sonunda varsayılan olarak `models/ocr_cnn.pt` kaydedilir (sınıf haritası dahil).

## Tek görüntüden çıkarım (segmentasyon + tahmin + braille)

```bash
python3 scripts/predict_image.py --model models/ocr_cnn.pt --image path/to/text_image.png
```

## Arayüz (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Arayüz:
- Görüntü yükler
- Segmentlenen karakterleri ve tahmini metni gösterir
- Metni Braille çıktısına çevirip gösterir

## Notlar / Sınırlar (bilinçli)

- Segmentasyon, temiz arka plan/kontrast varsayımıyla çalışan **basit** bir yaklaşımdır (bağlantılı bileşenler).
- Bu prototip, “tam OCR” yerine **karakter sınıflandırma + basit segmentasyon** hedefler.
- Braille dönüşümü **kural tabanlıdır** (Grade-1, A–Z + 0–9 + temel noktalama).

