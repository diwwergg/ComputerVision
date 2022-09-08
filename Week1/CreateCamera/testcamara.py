import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    cv2.imshow('frame1', frame)
    cv2.waitKey(1)