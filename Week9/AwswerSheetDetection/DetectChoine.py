import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import imutils
from skimage import measure, morphology


def check_small_rectangles_centroid(contours, img_contours,area_size=800):
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_size:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img_contours, (cx, cy), 3, (0, 0, 255), -1)
                centroid = (cx, cy)
                centroids.append(centroid)
    return centroids, img_contours

def puttext_on_image(img, text, x, y):

    cv2.putText(img, text, (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return img


sizepaper = (widthImg, heightImg) = (500, 700)
imageResult = cv2.imread('cropper.jpg')
ut.initializetrackbars()
while True:
    imgGray = cv2.cvtColor(imageResult,cv2.COLOR_BGR2GRAY)
    imgGray = cv2.GaussianBlur(imgGray, (7, 7), 0)
    """ imgGray = cv2.bilateralFilter(imgGray, d=5,sigmaColor=10,sigmaSpace=10) #Blur the image for better edge detection """
    imgThresh = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=13,C=3)
    thres, area_size = ut.valtrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThresholdCanny = cv2.Canny(imgGray, thres[0], thres[1])
    kernel = np.ones((2, 2))
    imgDial = cv2.dilate(imgThresholdCanny, np.ones((2, 2)), iterations=2)  # APPLY DILATION
    imgThresholdCanny2 = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    # cv2.imshow("Canny", imgThresholdCanny)
    # cv2.imshow("imgDial", imgDial)
    # cv2.imshow("imgThresholdCanny2", imgThresholdCanny2)
    # cv2.waitKey(1)
    
    
    
     ## FIND ALL COUNTOURS
    imgContours = imageResult.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = imageResult.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThresholdCanny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    centroids, imgContours = check_small_rectangles_centroid(contours, imgContours, area_size)
    
    # sort the centroids by y coordinate
    if len(centroids) > 0 and len(centroids) == 6:
        centroids = sorted(centroids, key=lambda x: x[1])
        if centroids[2][0] < centroids[3][0]:
            w1 = centroids[2]
            w2 = centroids[3]
        else:
            w1 = centroids[3]
            w2 = centroids[2]
        ycenter = ((((w2[0]-w1[0]) // 2)+w1[0]) , w1[1])
        htopcenter = (ycenter[0] ,(ycenter[1]-centroids[0][1])// 2 + centroids[0][1] -10)
        hbelowcenter = (ycenter[0], (centroids[-1][1]-ycenter[1])// 2 +ycenter[1]+70)
        tophleft = (w1[0]+10, htopcenter[1])
        belowleft = (w1[0]+10, hbelowcenter[1])
        wrapPoints = (tophleft, hbelowcenter, hbelowcenter, belowleft )
        
        cv2.circle(imgContours, w1, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, 'w1', w1[0], w1[1])
        cv2.circle(imgContours, w2, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "w2", w2[0], w2[1])
        cv2.circle(imgContours, ycenter, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "ycenter", ycenter[0], ycenter[1])
        cv2.circle(imgContours, htopcenter, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "htop", htopcenter[0], htopcenter[1])
        cv2.circle(imgContours, hbelowcenter, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "hbelow", hbelowcenter[0], hbelowcenter[1])
        cv2.circle(imgContours, tophleft, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "tophleft", tophleft[0], tophleft[1])
        cv2.circle(imgContours, belowleft, 3, (255, 255, 0), -1)
        imgContours = puttext_on_image(imgContours, "belowleft", belowleft[0], belowleft[1])
        
        
        # checkimg = ut.four_point_transform(imgBigContour, wrapPoints)
        gray = cv2.cvtColor(imgBigContour, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=3)
        cv2.imshow("thresh", thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
    Choine = (120, 145, 170, 195, 220) # A, B, C, D, E
    
    # cv2.imwrite("choine.jpg", imgContours)
    cv2.imshow("Contours", imgContours)
    cv2.waitKey(1)