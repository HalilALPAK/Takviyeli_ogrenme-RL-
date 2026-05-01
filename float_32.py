"""from ultralytics import YOLO

# Kendi eğittiğin .pt modelini yükle
model = YOLO('fire.pt')

# OpenCV uyumlu olması için bu parametreler ŞART:
model.export(
    format='onnx', 
    imgsz=640, 
    opset=12,         # OpenCV 4.x sürümleri için en kararlı opset sürümü
    simplify=True     # Gereksiz katmanları birleştirerek OpenCV'nin işini kolaylaştırır
)"""
from ultralytics import YOLO

model = YOLO('fire.pt')

# Standart Float32 TFLite dönüşümü
model.export(
    format='tflite',
    imgsz=640,
    int8=False, # INT8 kapalı (Varsayılan zaten False)
)