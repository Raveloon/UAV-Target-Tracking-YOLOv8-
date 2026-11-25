from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)#For WebCam
cap.set(3,1280)
cap.set(4,720)

#cap = cv2.VideoCapture("../Videos/bikes.mp4")#For Video


model = YOLO("../Yolo-Weights/yolov8l.pt")
classNames = ["person","bicycle","car","motorbike","aeroplane", "bus",
              "train","truck","boat","traffic light", "fire hydrant",
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
while True:
    success, img = cap.read()
    result = model(img, stream= True)
    for r in result:
        boxes = r.boxes
        for box in boxes:

            #Rectangle
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


            #Conf
            conf = math.ceil(box.conf[0]*100)/100

            #ClassName
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            #Only-Class
            if currentClass =="person" and conf > 0.3:
                # Output
                cvzone.putTextRect(img, f"{classNames[cls]}{conf}", (max(0, x1), max(35, y1)), scale=1.7, thickness=2,offset=5)
                cvzone.cornerRect(img, (x1, y1, w, h), l=10)


            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #print(x1, y1, x2, y2)



    cv2.imshow("Image",img)
    cv2.waitKey(1)