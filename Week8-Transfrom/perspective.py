import cv2
import numpy as np
size = (300, 300)
pts1 = []
pts2 = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
def onClick(e, x, y, p, f):
    if e == cv2.EVENT_LBUTTONDOWN:
        pts1.append([x, y])
        print(pts1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onClick)
while True:
    _, frame = cap.read()
    if len(pts1) == 4:
        T = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
        frame2 = cv2.warpPerspective(frame, T, size)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours:
            center, r = cv2.minEnclosingCircle(ct)
            print(r)
        cv2.imshow('frame2', bin)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
