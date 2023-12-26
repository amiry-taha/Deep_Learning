#Realisé par TAHA AMIRY EMSI 2023/2024
# YOLO : "You Only Look Once" real time object detection
# YOLOv8 model trained on the Microsoft COCO dataset
# COCO : Common Objects In Context
import cv2
from ultralytics import YOLO

#load yolov8 model
model = YOLO('yolov8n.pt')

#load video

# video_path = './test.mp4'
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

ret = True

#read frames



while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
    cv2.imshow('frame', frame_)
    if cv2.waitKey(60) & 0xFF == ord('q'): break