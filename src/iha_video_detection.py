import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from simple_pid import PID

# ── DEVICE: GPU varsa CUDA, yoksa CPU ────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanilan cihaz: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")  

# Gösterim penceresi
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# ============================================================
#  DOSYA YOLLARI
# ============================================================
VIDEO_PATH  = os.environ.get("UAV_VIDEO_PATH", r"C:\Users\Emre\Desktop\İTÜ ATA Takımı Uçuş Kanıt Videosu  712619  TEKNOFEST Savaşan İHA 2023.mp4")
OUTPUT_PATH = os.environ.get("UAV_OUTPUT_PATH", "output_tracking.mp4")
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"HATA: Video acilamadi: {VIDEO_PATH}")
    exit()

# Gerçek boyut
ret, first_frame = cap.read()
if not ret:
    print("HATA: Video okunamadi.")
    exit()
frame_height, frame_width = first_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fps_video    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {frame_width}x{frame_height} @ {fps_video:.1f} FPS | Toplam: {total_frames} kare")

# Video kaydedici
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (frame_width, frame_height))
print(f"Cikti: {OUTPUT_PATH}")

# YOLO — model ve imgsz seçimi performansı doğrudan etkiler
# Hız sırası: yolov8n > yolov8s > yolov8m > yolov8l > yolov8x
# yolov8n: en hızlı (~6MB) | yolov8m: orta denge | yolov8l: en yavaş (eski)
model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")   # 'l' -> 'n': ~10x hız artışı!
YOLO_INPUT_SIZE = 640   # 1280 -> 640: ~4x hız artışı (küçük nesneler için 960 dene)

# DeepSORT — max_age yüksek: kaybolunca uzun süre tahmin eder
tracker = DeepSort(max_age=20, nn_budget=100)

# PID
center_x = frame_width  // 2
center_y = frame_height // 2
pid_x = PID(0.2, 0.01, 0.1, setpoint=center_x)
pid_y = PID(0.2, 0.01, 0.1, setpoint=center_y)
pid_x.output_limits = (-300, 300)
pid_y.output_limits = (-300, 300)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus",
              "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
              "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork",
              "knife", "spoon", "bowl", "banana", "apple", "sandwich",
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
              "cake", "chair", "sofa", "pottedplant", "bed", "diningTable",
              "toilet", "tvMonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

TARGET_CLASSES = ["person", "car", "motorbike", "aeroplane", "truck", "bus", "boat"]

# Sınıfa özel confidence eşiği
# aeroplane için düşük eşik → küçük/uzak İHA kaçırılmasın
CONF_THRESHOLD = {
    "aeroplane": 0.25,   # düşük → uzak İHA yakalanır
    "default":   0.45,
}

# CLAHE — kontrast artışı (küçük nesneleri ön plana çıkarır)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Son bilinen konum (kayıp durumunda göstermek için)
last_known = {}          # track_id → (x1,y1,x2,y2, label, age)
LOST_SHOW_FRAMES = 30   # Kaybolunca kaç kare daha göster

# CLAHE her karede değil, N karede bir uygulanır → CPU yükü azalır
CLAHE_INTERVAL = 3   # Her 3. karede bir CLAHE uygula

prev_time   = time.time()
frame_count = 0
img_enhanced_cache = None   # CLAHE sonucunu önbelleğe al

print("'q'=cikis   'p'=duraklat")

cv2.namedWindow("IHA - Goruntu Isleme", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Video bitti.")
        break

    frame_count += 1
    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    # ── 1. CLAHE Ön İşleme (her CLAHE_INTERVAL karede bir) ─────────────
    if frame_count % CLAHE_INTERVAL == 1 or img_enhanced_cache is None:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe  = clahe.apply(l)
        img_enhanced_cache = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    img_enhanced = img_enhanced_cache

    # ── 2. YOLO Tespiti (GPU'da) ──────────────────────────
    results    = model(img_enhanced, stream=True, imgsz=YOLO_INPUT_SIZE, device=DEVICE, verbose=False)
    detections = []
    yolo_data  = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = list(map(int, box.xyxy[0].tolist()))
            x1 = max(0, min(x1, frame_width  - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width  - 1))
            y2 = max(0, min(y2, frame_height - 1))

            conf = round(float(box.conf[0]), 2)
            cls  = int(box.cls[0])
            name = classNames[cls]

            # Sınıfa özel eşik
            threshold = CONF_THRESHOLD.get(name, CONF_THRESHOLD["default"])

            bw, bh = x2 - x1, y2 - y1
            if (name in TARGET_CLASSES
                    and conf > threshold
                    and bw > 15 and bh > 15          # küçük nesneler için gevşetildi
                    and bw < frame_width  * 0.65
                    and bh < frame_height * 0.65):
                detections.append(([x1, y1, x2, y2], conf, cls))
                yolo_data.append((x1, y1, x2, y2, conf, cls))

    # ── 3. DeepSORT Takip ────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=img)

    # Track → en yakın YOLO kutusu eşleştir
    used_yolo     = set()
    track_to_yolo = {}
    active_ids    = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        active_ids.add(track.track_id)
        tx1, ty1, tx2, ty2 = track.to_ltrb()
        t_cx = (tx1 + tx2) / 2
        t_cy = (ty1 + ty2) / 2

        best_idx  = None
        best_dist = float('inf')
        max_dist  = math.sqrt(frame_width**2 + frame_height**2) * 0.30

        for idx, (x1, y1, x2, y2, conf, cls) in enumerate(yolo_data):
            if idx in used_yolo:
                continue
            d = math.sqrt(((x1+x2)/2 - t_cx)**2 + ((y1+y2)/2 - t_cy)**2)
            if d < best_dist and d < max_dist:
                best_dist = dist = d
                best_idx  = idx

        if best_idx is not None:
            track_to_yolo[track.track_id] = best_idx
            used_yolo.add(best_idx)

    # ── 4. Görselleştirme ─────────────────────────────────
    first_track = True

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id

        if tid in track_to_yolo:
            # ✅ Aktif tespit var → YOLO kutusuyla çiz
            x1, y1, x2, y2, conf, cls_id = yolo_data[track_to_yolo[tid]]
            label = classNames[cls_id]

            # Son bilinen konumu güncelle
            last_known[tid] = [x1, y1, x2, y2, label, 0]

            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2

            control_x = pid_x(target_x)
            control_y = pid_y(target_y)
            dist_val  = math.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)

            dir_x = "SOL"    if control_x < -5 else ("SAG"   if control_x > 5 else "MERKEZ")
            dir_y = "YUKARI" if control_y < -5 else ("ASAGI" if control_y > 5 else "MERKEZ")

            # Kutu + etiket
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1),
                              l=20, rt=3,
                              colorR=(0, 220, 255), colorC=(0, 220, 255))
            cvzone.putTextRect(img,
                               f"{label} #{tid}  {dist_val:.0f}px  [{conf:.2f}]",
                               (max(0, x1), max(35, y1)),
                               scale=1.1, thickness=1, offset=5,
                               colorR=(160, 0, 220), colorT=(255, 255, 255))

            # Hedef merkez
            cv2.circle(img, (target_x, target_y), 7,  (0, 0, 255), -1)
            cv2.circle(img, (target_x, target_y), 13, (0, 0, 255), 2)
            cv2.line(img, (center_x, center_y), (target_x, target_y),
                     (0, 255, 255), 2)

            if first_track:
                cv2.putText(img, f"PID X:{control_x:+.1f}  Y:{control_y:+.1f}",
                            (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(img, f"Yon: {dir_x} / {dir_y}",
                            (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(img, f"Uzaklik: {dist_val:.1f} px",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                first_track = False

        else:
            # ⚠️ Tespit yok ama track hâlâ canlı → "ARANIYOR" kutusu göster
            if tid in last_known:
                lx1, ly1, lx2, ly2, llabel, age = last_known[tid]
                age += 1
                last_known[tid][5] = age

                if age <= LOST_SHOW_FRAMES:
                    alpha = max(0.2, 1.0 - age / LOST_SHOW_FRAMES)
                    # Kesikli sarı kutu
                    overlay = img.copy()
                    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 200, 255), 2)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                    # "ARANIYOR" etiketi
                    cvzone.putTextRect(img,
                                       f"ARANIYOR: {llabel} #{tid}",
                                       (max(0, lx1), max(35, ly1)),
                                       scale=1.0, thickness=1, offset=4,
                                       colorR=(0, 100, 255), colorT=(255, 255, 255))

    # Eski kayıp track'leri temizle
    for tid in list(last_known.keys()):
        if tid not in active_ids:
            last_known[tid][5] += 1
            if last_known[tid][5] > LOST_SHOW_FRAMES * 2:
                del last_known[tid]

    # Crosshair
    cv2.drawMarker(img, (center_x, center_y), (0, 60, 255),
                   cv2.MARKER_CROSS, 40, 2)

    # FPS / kare
    cv2.putText(img, f"FPS: {fps:.1f}  Kare: {frame_count}/{total_frames}",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── 5. Kaydet ────────────────────────────────────────
    out.write(img)

    if frame_count % 100 == 0:
        pct = frame_count / total_frames * 100
        print(f"  Isleniyor: {frame_count}/{total_frames}  ({pct:.1f}%)")

    # ── 6. Ekrana göster ─────────────────────────────────
    h, w   = img.shape[:2]
    scale  = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
    disp   = cv2.resize(img, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR) if scale < 1.0 else img

    cv2.imshow("IHA - Goruntu Isleme", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Cikis.")
        break
    elif key == ord('p'):
        print("Duraklatildi...")
        cv2.waitKey(0)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tamamlandi! Cikti: {OUTPUT_PATH}")
