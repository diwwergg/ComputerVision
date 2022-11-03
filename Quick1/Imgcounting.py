import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from Quick1 import util1 as ut

img = cv2.imread("6352300197.png")

def nothing(x):
    pass
def initializetrackbars(intialTracbar_vals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("AreaSize", "Trackbars", 50, 1000, nothing)

def valtrackbars():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = (threshold1, threshold2)
    return src

def check_small_rectangles_centroid(contours, img_contours):
    centroids = []
    contours_after = []
    counter = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) > 6:
            contours_after.append(cnt)
            counter = counter + 1
    return contours_after, img_contours, counter


plt.figure('org')
plt.imshow(img)


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure('gray')
plt.imshow(imgGray, cmap='gray')

imgThresh = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=3,C=2)
plt.figure('imgThresh')
plt.imshow(imgThresh, cmap='gray')

canny = cv2.Canny(img, 20, 15)
dilated = cv2.dilate(canny, (1.2, 1.2), iterations=1)
plt.figure('canny')
plt.imshow(canny)
plt.figure('dilate')
plt.imshow(dilated)
plt.show()


cnt, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_after, img_contours, counter = check_small_rectangles_centroid(cnt, canny)
print(counter)

# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb2 = cv2.drawContours(img, contours_after, -1, (0, 255, 0), 2)
plt.figure('counter')
plt.imshow(img)
plt.show()
