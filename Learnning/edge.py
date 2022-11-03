import cv2
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    frame = frame[:frame.shape[0] // 2, frame.shape[1]//2 + 20:, :]
    frame0 = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    temp = cv2.Canny(frame, 100, 200)
    s = np.sum(temp[:temp.shape[0]//2, :])
    text = 'paper' if s > 0 else 'rock'
    cv2.putText(frame0, text, (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 1)
    cv2.imshow('Canny', frame0)
    cv2.waitKey(1)