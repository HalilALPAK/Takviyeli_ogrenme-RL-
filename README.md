# Otonom Oyun Ajanı: Bilgisayarlı Görü ve Pekiştirmeli Öğrenme

Bu proje, bir oyunda (büyük ihtimalle Chicken Invaders veya benzeri bir uzay vurma oyunu) **Bilgisayarlı Görü (Computer Vision)** ve **Pekiştirmeli Öğrenme (Reinforcement Learning)** kullanarak kendi kendine hayatta kalabilen ve oynayabilen otonom bir yapay zeka ajanı geliştirmeyi amaçlamaktadır.

Sistem iki ana bileşenden oluşmaktadır:
1. **C++ İstemcisi**: Ekran görüntüsünü alır, nesne tespiti yapar (YOLO/ONNX) ve klavye hareketlerini simüle eder.
2. **Python Sunucusu**: C++ istemcisinden gelen ortam (grid) verisini alır, Derin Q-Ağı (DQN) kullanarak bir eylem seçer ve C++ istemcisine geri gönderir.

## 🏗️ Mimari ve Çalışma Mantığı

### 1. Görüntü İşleme ve Kontrol (C++ - `main.cpp`)
- **Ekran Yakalama**: Windows API (`BitBlt`) kullanılarak oyunun bulunduğu belirli bir ekran bölgesi (100, 100, 700x500 boyutlarında) saniyede birçok kez yakalanır.
- **Nesne Tespiti (YOLO)**: Yakalanan görüntüler 640x640 boyutuna getirilerek ONNX Runtime üzerinden eğitilmiş olan `tavuk.onnx` nesne tespit modeline sokulur.
- **Sınıflandırma**: Model ekrandaki Gemi (Ship), Düşman (Enemy/Asteroid), Mermi (Bullet) ve Ödül (Reward/Coin) gibi objeleri tespit eder.
- **Grid Haritalama**: Ekran pikselleri 8x10'luk bir ızgaraya (grid) bölünür ve tespit edilen nesnelerin hangi hücrede olduğu hesaplanarak JSON formatında Python sunucusuna (TCP 5001 portu) iletilir.
- **Aksiyon Uygulama**: Python'dan gelen `U, D, L, R, S` (Yukarı, Aşağı, Sol, Sağ, Ateş) komutlarını Windows API (`SendInput`) ile sanal klavye tuş vuruşlarına çevirir. Grid tabanlı hareket için tuş basılı tutma süresi 130ms olarak optimize edilmiştir.

### 2. Pekiştirmeli Öğrenme Ajanı (Python - `lr.py`)
- **Durum (State) Temsili**: C++'tan gelen JSON verisi 8x10'luk sayısal bir matrise çevrilir. (Gemi: 1, Düşman: -1, Mermi: -2, Ödül: 2).
- **Ajan (DQN)**: 80 girişli (8x10 grid) ve 6 çıkışlı (Eylemler) bir Yapay Sinir Ağı (Deep Q-Network) kullanır. Ajan ortamı deneyimledikçe `ci_model.pth` dosyasına ağırlıklarını kaydeder.
- **Deneyim Tekrarı (Replay Buffer)**: Ajanın geçmiş deneyimleri (Durum, Eylem, Ödül, Yeni Durum) hafızada tutulur ve rastgele partiler (batch) seçilerek ağ eğitilir.

## 🎯 Ödül Sistemi (Reward System)
Ajanın doğru kararlar verebilmesi için özel bir ödül ve ceza sistemi tasarlanmıştır:
- **Hayatta Kalma (Step Reward)**: Her başarılı adımda ajan **+0.1** ödül alır.
- **Ölüm / Gemi Kaybı**: Gemi ekrandan kaybolduğunda ajan **-20.0** ceza alır ve canı azalır.
- **Sınır İhlali**: Gemi oyun sınırlarının dışına çıkmaya çalıştığında boş yere hareket etmesini engellemek için **-1.0** ceza alır.

## 📂 Proje Dosyaları
* `main.cpp`: Bilgisayarlı görü, nesne algılama ve klavye kontrol kodu (C++).
* `lr.py`: Derin Q-Ağı (DQN) modelini eğiten ve kararları veren sunucu kodu (Python).
* `tavuk.onnx`: ONNX formatında eğitilmiş YOLO nesne tanıma modeli.
* `ci_model.pth`: Python RL ajanının eğitilmiş PyTorch ağırlıkları.

## 🚀 Gereksinimler ve Kurulum

### C++ Bağımlılıkları
- OpenCV (Görüntü işleme ve hata ayıklama penceresi için)
- ONNX Runtime (C++ API, `tavuk.onnx` modelini çalıştırmak için)
- Winsock2 & Windows API (Socket haberleşmesi ve klavye simülasyonu)

### Python Bağımlılıkları
```bash
pip install torch numpy
```

## 🎮 Nasıl Çalıştırılır?
1. **Oyunu Açın**: Hedef oyunu (örn. Chicken Invaders) ekranın sol üst köşesine (100, 100) gelecek şekilde ayarlayın (Çözünürlük ~700x500).
2. **Python Sunucusunu Başlatın**: İlk olarak RL ajanını çalıştırın ve TCP portunu dinlemeye alın.
   ```bash
   python lr.py
   ```
3. **C++ İstemcisini Başlatın**: `main.exe` programını çalıştırarak ekran okumayı ve modeli aktifleştirin. İstemci, Python'a bağlandığı anda oyun otonom olarak oynanmaya başlayacaktır.
4. **İzleyin**: "AI Vision Debug" adlı OpenCV penceresinde tespit edilen nesneleri, Python terminalinde ise ajan kararlarını, anlık ödül/cezaları ve 8x10'luk çevrilmiş grid dünyasını izleyebilirsiniz.
