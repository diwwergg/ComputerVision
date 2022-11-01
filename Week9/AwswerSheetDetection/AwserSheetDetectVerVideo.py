from ctypes import pointer
from tkinter import image_names
from turtle import right
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology
# import utils file from Week9
import utils as ut

sizepaper = (500,700)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    cv2.imshow("Original", frame)
    image = frame.copy()
    #Resize the image for better and fast processing
    r = 600.0 / image.shape[1]
    dim = (int(image.shape[1]*r),1000)
    # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, dim)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 5, 21, 21) #Blur the image for better edge detection
    #Edge detection using canny algorithm
    edged = cv2.Canny(image,30,160)
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    img = image.copy()
    cv2.drawContours(img,contours,-1,color=(255, 255, 255),thickness = 2)
    for c in contours:
        arc_length = 0.1 * cv2.arcLength(c,True)  #Find the perimeter of closed curve 10%
        no_of_points_found = cv2.approxPolyDP(c,arc_length,True)

        if len(no_of_points_found) == 4:
            Main_contour = no_of_points_found
            break
    copy = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img,[Main_contour],-1,(0,255,0),2)
    cv2.imshow("Main outline",img)


    # Set Pointer Rectengle
    topleft = Main_contour[1][0]
    topRight = Main_contour[0][0]
    bottomLeft = Main_contour[2][0]
    bottomRight = Main_contour[3][0]
    pointers = (topleft,topRight,bottomLeft,bottomRight)
    pointers1 = np.float32([topleft,topRight,bottomLeft,bottomRight])

    pointers2 = np.float32([[0, 0],[sizepaper[0], 0],[0, sizepaper[1]],[sizepaper[0],sizepaper[1]]])

    matrix = cv2.getPerspectiveTransform(pointers1,pointers2)
    cropper = cv2.warpPerspective(image,matrix,sizepaper)
    cv2.imshow("Cropped",cropper)
    cv2.waitKey(1)
