import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

print(face_classifier)
cap = cv2.VideoCapture('ped.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

        cap.release()
        cv2.destroyAllWindows()
