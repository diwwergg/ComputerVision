import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
bg = None
iframe = 0
while True:
    iframe += 1
    _, frame = cap.read()
    if iframe == 100:
        bg = frame.copy()
    if bg is not None:
        diff = cv2.absdiff(bg, frame)
        cv2.imshow('diff', diff)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)