from ultralytics import YOLO

# Kendi eğittiğin .pt modelini yükle
model = YOLO('tavuk_yolo.pt')

# OpenCV uyumlu olması için bu parametreler ŞART:
model.export(
    format='onnx', 
    imgsz=640, 
    opset=16, 
    simplify=True,
    half=True  # Bu parametre modeli Float16 yapar
)
"""
from ultralytics import YOLO

model = YOLO('best2.pt')

# Float16 TFLite dönüşümü
model.export(
    format='tflite',
    imgsz=640,
    half=True, # Bu parametre Float16 (Yarı hassasiyet) yapar
    name='kiyafet'
)"""