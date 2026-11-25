import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from simple_pid import PID

cap = cv2.VideoCapture(0)  # For WebCam
cap.set(3, 1920)
cap.set(4, 1080)

# YOLO Model
model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")


# DeepSORT Tracker
tracker = DeepSort(max_age=1)

# PID Controller
frame_width = 1280
frame_height = 720
center_x, center_y = frame_width // 2, frame_height // 2
pid_x = PID(0.2, 0.01, 0.1, setpoint=center_x)  # Kamera genişliği ortası
pid_y = PID(0.2, 0.01, 0.1, setpoint=center_y)  # Kamera yüksekliği ortası

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

prev_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model(img, stream=True)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = list(map(int, box.xyxy[0].tolist()))
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                detections.append(([x1, y1, x2, y2], conf, cls))

    # DeepSORT ile takip
    tracks = tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        target_x, target_y = (x1 + x2) // 2, (y1 + y2) // 2

        # PID kontrol
        control_x = pid_x(target_x)
        control_y = pid_y(target_y)

        # Uzaklık hesabı
        distance_x = center_x - target_x
        distance_y = center_y - target_y
        total_distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        # PID değişim isteği bilgisi
        pid_output_text = f"PID Output - X: {control_x:.2f}, Y: {control_y:.2f}"
        distance_text = f"Distance: {total_distance:.2f} px"

        print(pid_output_text)
        print(distance_text)

        # Görselleştirme
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Daha kesin kutu çizimi
        cvzone.putTextRect(img, f"Person {conf}", (max(0, x1), max(35, y1)), scale=1.7, thickness=2, offset=5)
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=3)
        cv2.circle(img, (target_x, target_y), 5, (0, 0, 255), -1)
        cv2.putText(img, pid_output_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, distance_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS gösterimi
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(img, fps_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Ekranın orta noktasını işaretle
    cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
