import os

import torch
import cv2
from ultralytics import YOLO
import time

# Cihaz (GPU/CPU) seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO modelini yükle (Dosyanın doğru yolunu belirttiğinden emin ol)
model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt").to(device)

# Webcam'i aç
cap = cv2.VideoCapture(0)  # Eğer test görüntüsü kullanacaksan, burada cap.read() ile görüntüyü al
frame_rate = 0
prev_time = 0

while True:
    # Webcam'den bir frame al
    ret, frame = cap.read()
    if not ret:
        break  # Eğer video biterse çık

    # RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Giriş boyutunu 640x640 olarak ayarla
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    # Tensör haline getir (BCHW formatına çevir)
    img_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Zaman hesaplaması
    start_time = time.time()

    # Modeli çalıştır
    results = model(img_tensor)

    # FPS hesaplama
    end_time = time.time()
    exec_time = end_time - start_time
    fps = 1 / exec_time if exec_time > 0 else 0

    # Sonuçları manuel olarak işleyelim
    # Çıktıdaki tespitleri al
    pred = results[0]  # Modelin ilk sonucu
    boxes = pred.boxes.xywh  # Kutu koordinatları (xywh)
    confs = pred.boxes.conf  # Güven skorları
    classes = pred.boxes.cls  # Sınıflar

    # Tespitleri çizmeye başla
    for i in range(len(boxes)):
        # Kutu koordinatlarını al (xmin, ymin, xmax, ymax)
        x, y, w, h = boxes[i]
        x1 = int((x - w / 2) * frame.shape[1])
        y1 = int((y - h / 2) * frame.shape[0])
        x2 = int((x + w / 2) * frame.shape[1])
        y2 = int((y + h / 2) * frame.shape[0])

        # Güven skorunu al
        conf = confs[i].item()

        # Tespit edilen sınıf ismini al
        class_id = int(classes[i].item())
        class_name = model.names[class_id]  # model.names, sınıf adlarını içeren bir liste

        # Kutu ve etiket çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS değerini ekrana yazdır
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Görüntüyü ekrana göster
    cv2.imshow('YOLO Object Detection', frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
