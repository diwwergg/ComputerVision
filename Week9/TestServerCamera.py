import cv2

cap = cv2.VideoCapture('rtsp://172.23.45.74:8080/h264_ulaw.sdp')
while True:
    _, frame = cap.read()
    img = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('img', img)
    cv2.waitKey(1)