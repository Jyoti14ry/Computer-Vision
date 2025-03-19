import datetime
import tkinter as tk
from tkinter import filedialog
import cv2
import os
debug = True
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_face_from_video():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file = os.path.join(
        "output/video", f"output_video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.avi")
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
    if debug:
        print("Video capture started")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        out.write(frame)
        cv2.imshow('Face Detection from Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if debug:
        print("Video capture completed")
detect_face_from_video()
