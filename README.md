# Autonomous UAV Target Tracking & Gimbal Locking System (YOLOv8 + DeepSORT + PID)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20(Ultralytics)-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![DeepSORT](https://img.shields.io/badge/DeepSORT-Real--Time%20MOT-4B8BBE)](https://github.com/levan92/deep_sort_realtime)
[![TEKNOFEST](https://img.shields.io/badge/Competition-TEKNOFEST%20Sava%C5%9Fan%20%C4%B0HA-red)](https://www.teknofest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and closed-loop control system developed for the **TEKNOFEST Combat UAV (Savaşan İHA)** competition. The pipeline integrates real-time object detection (**YOLOv8**), multi-target persistence tracking (**DeepSORT**), and a 2-axis closed-loop **PID controller** designed to center and lock onto airborne targets with a servo-driven Pan-Tilt gimbal camera.

---

## 🌟 System Architecture

```mermaid
flowchart LR
    subgraph Video Input
        Cam[Camera / UAV Flight Stream]
    end

    subgraph Deep Learning Inference
        Cam -->|Frames (640x640)| YOLO[YOLOv8 Deep Neural Network<br/>CUDA Accelerated]
        YOLO -->|Bounding Boxes & Confidences| DeepSORT[DeepSORT Tracker<br/>Kalman Filter + ReID]
    end

    subgraph Control Loop
        DeepSORT -->|Target Center (x, y)| PID[Dual-Axis PID Controller<br/>Pan-Tilt Error Correction]
        PID -->|Control Signal (ΔPan, ΔTilt)| Actuator[Gimbal / Servo Motor Actuators]
    end

    subgraph HUD & Telemetry
        DeepSORT --> Vis[Real-Time HUD Telemetry<br/>OpenCV & CVZone Display]
        PID --> Vis
    end
```

---

## 🚀 Key Modules & Capabilities

### 🔍 1. Real-Time Aerial Detection (YOLOv8)
- GPU-accelerated inference utilizing PyTorch CUDA with automatic CPU fallback.
- Native input resolution scaling (640×640) achieving up to **10x higher FPS** compared to unscaled processing while retaining small target resolution.
- Multi-tier model support: `yolov8n` (ultra-fast nano for embedded flight computers), `yolov8m`, and `yolov8l` (high-accuracy ground stations).

### 🎯 2. Multi-Object Tracking & Re-Identification (DeepSORT)
- Combines recursive **Kalman Filtering** for state trajectory estimation with deep appearance descriptors (ReID) to maintain persistent target IDs.
- Robust target recovery across occlusion, rapid turns, and background clutter with parameterized `max_age=20` and nearest-neighbor budget limits (`nn_budget=100`).

### 🎮 3. Closed-Loop PID Gimbal Controller
- Computes real-time 2D pixel distance error between the locked target center $(x_t, y_t)$ and the camera optical center $(x_c, y_c)$:
  $$e_x = x_t - x_c, \quad e_y = y_t - y_c$$
- Independent proportional-integral-derivative controllers compute continuous motor velocity adjustments:
  $$u(t) = K_p e(t) + K_i \int_0^t e(	au)d	au + K_d rac{de(t)}{dt}$$
- Dampens overshoot and eliminates mechanical hunting during agile aerial maneuvers.

### 📊 4. Tactical Flight Telemetry HUD
- Real-time on-screen telemetry rendering via OpenCV & CVZone:
  - Active Target ID and detection confidence.
  - Pixel error distance to crosshair center.
  - Live PID output vectors for X and Y axes.
  - Real-time inference FPS counter.
  - Automated output video recording (`VideoWriter`).

---

## 📂 Repository Structure

```
UAV-Target-Tracking-YOLOv8/
├── docs/
│   └── TEKNOFEST_KTR_Raporu.docx   # TEKNOFEST Critical Design Report (KTR)
├── models/
│   └── yolov8n.pt                  # Pre-trained Nano weights (ready out-of-the-box)
├── src/
│   ├── iha_video_detection.py      # Core UAV flight video processor with HUD & recording
│   ├── obj_det_deepsort_pid.py     # Live webcam / sensor tracking with PID controller
│   ├── yolo_tensor_inference.py    # Direct PyTorch BCHW tensor inference pipeline
│   └── yolo_webcam.py              # Lightweight webcam visualizer
├── requirements.txt                # Python environment dependencies
└── .gitignore
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA drivers installed (recommended for real-time 60+ FPS performance)

### 2. Environment Setup
```bash
git clone https://github.com/Raveloon/UAV-Target-Tracking-YOLOv8-.git
cd UAV-Target-Tracking-YOLOv8-
pip install -r requirements.txt
```

### 3. Running Flight Video Detection & Tracking
To process an aerial flight video with HUD overlay and export the annotated output:
```bash
python src/iha_video_detection.py
```
*(Press `q` to quit, `p` to pause/resume).*

### 4. Running Live Camera PID Tracking
To start real-time target locking and PID output computation with a local camera:
```bash
python src/obj_det_deepsort_pid.py
```

---

## 🏆 Competition Background

This system was engineered as part of the avionics and image processing sub-teams for **TEKNOFEST Savaşan İHA (Combat UAV)**. The goal was detecting opponent UAVs in contested airspace, establishing target locks, and feeding guidance vectors to the flight controller.

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).
