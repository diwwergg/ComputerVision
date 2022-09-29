import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    frame2 = frame[:frame.shape[0] // 2, frame.shape[1] // 2 + 20:, :]
    cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Canny', cv2.Canny(frame2, 100, 200))
    cv2.imshow('Canny', cv2.Canny(frame, 100, 200))
    cv2.waitKey(1)
