import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
prev_frame = None
L = 50
L_thick = 10
iframe = 0
while True:
    iframe += 1
    _, frame = cap.read()
    L_act = False
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame[:, L:L+L_thick], frame[:, L:L+L_thick])
        if diff.sum() > 80000:
            L_act = True
            print(iframe)
        cv2.imshow('diff', diff)
    prev_frame = frame.copy()
    if L_act:
        frame[:, L:L + L_thick, :2] = 0
    else:
        frame[:, L:L + L_thick, [0, 2]] = 0
    cv2.imshow('frame', frame)
    cv2.waitKey(1)