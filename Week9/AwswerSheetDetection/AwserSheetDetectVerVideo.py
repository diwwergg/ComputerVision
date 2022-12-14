import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils as ut

def find_big_rectangle(contours):
    check = False
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        arc_length = 0.15 * cv2.arcLength(c,True)  #Find the perimeter of closed curve 10%
        no_of_points_found = cv2.approxPolyDP(c,arc_length,True)
        if len(no_of_points_found) == 4:
            main_contour = no_of_points_found
            check = True
            break
    if check == True:
        return main_contour
    return []
def get_iamge_frame():
    if video:
        _, frame = cap.read()
        image = frame.copy()
    else:
        image = cv2.imread('papersheet4.jpg')
    return image

def resize_image(image):
    if image.shape[0] > image.shape[1]:
        r = 600.0 / image.shape[1]
        dim = (1000,int(image.shape[1]*r))
    else:
        r = 600.0 / image.shape[1]
        dim = (int(image.shape[1]*r), 1000)
    image = cv2.resize(image, dim)
    return image

def set_main_contour_pointer_rectengle(main_contour):
    topleft = main_contour[1][0]
    top_right = main_contour[0][0]
    bottom_left = main_contour[2][0]
    bottom_right = main_contour[3][0]
    pointers1 = np.float32([topleft,top_right,bottom_left,bottom_right])
    pointers2 = np.float32([[0, 0],[sizepaper[0], 0],[0, sizepaper[1]],[sizepaper[0],sizepaper[1]]])
    matrix = cv2.getPerspectiveTransform(pointers1,pointers2)
    return matrix


def check_small_rectangles(contours, area_size=800):
    rectengle_cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_size:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                rectengle_cnts.append(cnt)
    return rectengle_cnts
    
# sizepaper = (500,700)
sizepaper = (widthImg, heightImg) = (500, 700)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video = True

ut.initializetrackbars()
while True:
    image = get_iamge_frame()
    # image = resize_image(org)
    # cv2.imshow("Original", frame)
    copy = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 5, 21, 21) #Blur the image for better edge detection
    #Edge detection using canny algorithm
    edged = cv2.Canny(image,30,160)
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    img = image.copy()
    # cv2.drawContours(img,contours,-1,color=(255, 255, 255),thickness = 2)
    main_contour = find_big_rectangle(contours)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    if len(main_contour)!=0:
        cv2.drawContours(img,[main_contour],-1,(0,255,0),2)
        # Set Pointer Rectengle
        matrix_paper = set_main_contour_pointer_rectengle(main_contour)
        cropper_paper = cv2.warpPerspective(image,matrix_paper,sizepaper)
        cv2.imshow("Cropped",cropper_paper)
        imageResult = cropper_paper.copy()
        # WRAPPER SMALL RECTANGLES IN PAGE
        cv2.waitKey(1)
        cv2.imwrite('cropper.jpg', imageResult)

        
        
        
    img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
    cv2.imshow("Main outline",img)
    cv2.waitKey(1) 
