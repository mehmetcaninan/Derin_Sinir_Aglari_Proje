import numpy as np
import os
from collections import Counter
from PIL import Image

print("CIFAR-10 k-NN Sınıflandırma Algoritması")
print("=" * 50)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

data_path = "/Users/mehmetcaninan/Desktop/derin-sinir-aglari/cifar10"

print("CIFAR-10 veri seti yükleniyor...")

# Training verilerini yükleme
train_images = []
train_labels = []

print("Training verilerini yükleniyor...")
train_path = os.path.join(data_path, "train")

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(train_path, class_name)
    if not os.path.exists(class_path):
        print(f"Hata: {class_path} klasörü bulunamadı!")
        exit()

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"{class_name} sınıfı: {len(image_files)} görüntü")

    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        try:
            # Görüntüyü yükleme ve numpy array'e çevirme
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_array = np.array(img)

            # 32x32x3 boyutuna yeniden boyutlandırma (CIFAR-10 standart boyutu)
            if img_array.shape != (32, 32, 3):
                img = img.resize((32, 32))
                img_array = np.array(img)

            # Düzleştirme (32x32x3 = 3072)
            img_flattened = img_array.flatten()

            train_images.append(img_flattened)
            train_labels.append(class_idx)
        except Exception as e:
            print(f"Hata: {image_path} yüklenirken sorun oluştu: {e}")

# Training verilerini numpy array'e çevirme
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Test verilerini yükleme
test_images = []
test_labels = []

print("Test verilerini yükleniyor...")
test_path = os.path.join(data_path, "test")

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(test_path, class_name)
    if not os.path.exists(class_path):
        print(f"Hata: {class_path} klasörü bulunamadı!")
        exit()

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        try:
            # Görüntüyü yükleme ve numpy array'e çevirme
            img = Image.open(image_path)
            img = img.convert('RGB')  # RGB formatına çevirme
            img_array = np.array(img)

            # 32x32x3 boyutuna yeniden boyutlandırma
            if img_array.shape != (32, 32, 3):
                img = img.resize((32, 32))
                img_array = np.array(img)

            # Düzleştirme
            img_flattened = img_array.flatten()

            test_images.append(img_flattened)
            test_labels.append(class_idx)
        except Exception as e:
            print(f"Hata: {image_path} yüklenirken sorun oluştu: {e}")

# Test verilerini numpy array'e çevirme
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(f"\nTraining veri seti: {train_images.shape[0]} örnek")
print(f"Test veri seti: {test_images.shape[0]} örnek")
print(f"Görüntü boyutu: {train_images.shape[1]} piksel (32x32x3 = 3072)")

# Veriyi normalize etme (0-1 arasına getirme)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Kullanıcıdan mesafe metriği seçimi
print("\nMesafe metriği seçiniz:")
print("1. L1 (Manhattan) mesafesi")
print("2. L2 (Öklid) mesafesi")

while True:
    choice = input("Seçiminizi yapın (1 veya 2): ").strip()
    if choice == '1':
        distance_metric = 'L1'
        break
    elif choice == '2':
        distance_metric = 'L2'
        break
    else:
        print("Geçersiz seçim! Lütfen 1 veya 2 girin.")

# Kullanıcıdan k değeri alma
while True:
    try:
        k = int(input("k değerini girin (örnek: 3, 5, 7): "))
        if k > 0 and k <= len(train_images):
            break
        else:
            print(f"k değeri 1 ile {len(train_images)} arasında olmalıdır!")
    except ValueError:
        print("Geçersiz değer! Lütfen pozitif bir tam sayı girin.")

print(f"\nSeçilen mesafe metriği: {distance_metric}")
print(f"Seçilen k değeri: {k}")
print("\nSınıflandırma başlıyor...")

# Test için daha az örnek kullanarak hızlandırma
test_size = min(50, len(test_images))
print(f"Test edilecek örnek sayısı: {test_size}")

correct_predictions = 0

for test_idx in range(test_size):
    test_image = test_images[test_idx]
    true_label = test_labels[test_idx]

    # Tüm training örnekleri ile mesafe hesaplama
    distances = []

    for train_idx in range(len(train_images)):
        train_image = train_images[train_idx]

        # Mesafe hesaplama
        if distance_metric == 'L1':
            # Manhattan mesafesi
            distance = np.sum(np.abs(test_image - train_image))
        else:  # L2
            # Öklid mesafesi
            distance = np.sqrt(np.sum((test_image - train_image) ** 2))

        distances.append((distance, train_labels[train_idx]))

    # En yakın k komşuyu bulma
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    # K komşunun etiketleri
    neighbor_labels = [label for _, label in k_nearest]

    # Çoğunluk oylaması
    label_counts = Counter(neighbor_labels)
    predicted_label = label_counts.most_common(1)[0][0]

    # Doğru tahmin kontrolü
    if predicted_label == true_label:
        correct_predictions += 1

    # İlerleme gösterimi
    if (test_idx + 1) % 10 == 0 or test_idx < 5:
        print(f"Test örneği {test_idx + 1}/{test_size} - Gerçek: {class_names[true_label]}, Tahmin: {class_names[predicted_label]}")

    # İlk 3 örnek için detaylı sonuç gösterimi
    if test_idx < 3:
        print(f"  En yakın {k} komşu: {[class_names[label] for label in neighbor_labels]}")

# Sonuçları gösterme
accuracy = (correct_predictions / test_size) * 100
print(f"\n{'='*50}")
print(f"SONUÇLAR:")
print(f"Test edilen örnek sayısı: {test_size}")
print(f"Doğru tahmin sayısı: {correct_predictions}")
print(f"Doğruluk oranı: {accuracy:.2f}%")
print(f"Kullanılan mesafe metriği: {distance_metric}")
print(f"k değeri: {k}")
print(f"{'='*50}")

print("\nProgram tamamlandı.")

# Mehmet Can İnan - 02210224030