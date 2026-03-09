import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# -------------------------------
# 1️⃣ Modeli yükle
# -------------------------------
model = YOLO("best.pt")  # kendi model yolunu yaz

# -------------------------------
# 2️⃣ Ekran yakalama ayarları
# -------------------------------
monitor = {"top": 100, "left": 100, "width": 700, "height": 500}  # oyun penceresinin koordinatları
sct = mss()

# -------------------------------
# 3️⃣ Gerçek zamanlı tespit
# -------------------------------
while True:
    # Oyun penceresini yakala
    screen = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # BGR formatına çevir

    # -------------------------------
    # 4️⃣ Modeli çalıştır
    # -------------------------------
    results = model(frame)  # YOLOv8-style inference

    # -------------------------------
    # 5️⃣ Sonuçları frame üzerine çiz
    # -------------------------------
    annotated_frame = results[0].plot()  # kutular çizilir

    # Ekrana göster
    cv2.imshow("Chicken Invaders Detection", annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
