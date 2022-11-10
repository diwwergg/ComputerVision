import cv2
import matplotlib.pyplot as plt
import numpy as np

loop = True
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    if loop:
        ret, frame = cap.read()
    else:
        frame = cv2.imread('test.jpg')
        break
    cv2.imshow('frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('gray', gray)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    cv2.imshow('thresh', thresh)


    if cv2.waitKey(1) == ord('q'):
        break