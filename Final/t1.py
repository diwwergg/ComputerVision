
import numpy as np
import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

img = cv2.imread('water.jpg')
# img = cv2.resize(img, (640, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]

plt.imshow(img, cmap='gray')
plt.show()
