import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import imutils


def check_small_rectangles_centroid1(contours, img_contours, area_size=800):
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

def puttext_on_image1(img, text, x, y):

    cv2.putText(img, text, (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return img


def find_big_rectangle(contours):
    check = False
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        arc_length = 0.15 * cv2.arcLength(c, True)  # Find the perimeter of closed curve 10%
        no_of_points_found = cv2.approxPolyDP(c, arc_length, True)
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
        dim = (1000, int(image.shape[1] * r))
    else:
        r = 600.0 / image.shape[1]
        dim = (int(image.shape[1] * r), 1000)
    image = cv2.resize(image, dim)
    return image


def set_main_contour_pointer_rectengle(main_contour):
    topleft = main_contour[1][0]
    top_right = main_contour[0][0]
    bottom_left = main_contour[2][0]
    bottom_right = main_contour[3][0]
    pointers1 = np.float32([topleft, top_right, bottom_left, bottom_right])
    pointers2 = np.float32([[0, 0], [sizepaper[0], 0], [0, sizepaper[1]], [sizepaper[0], sizepaper[1]]])
    matrix = cv2.getPerspectiveTransform(pointers1, pointers2)
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



sizepaper = (widthImg, heightImg) = (500, 700)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video = True
ut.initializetrackbars()
while True:
    image = get_iamge_frame()
    # image = resize_image(org)
    # cv2.imshow("Original", frame)
    copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 5, 21, 21)  # Blur the image for better edge detection
    # Edge detection using canny algorithm
    edged = cv2.Canny(image, 30, 160)
    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    img = image.copy()
    # cv2.drawContours(img,contours,-1,color=(255, 255, 255),thickness = 2)
    main_contour = find_big_rectangle(contours)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(main_contour) != 0:
        cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)
        # Set Pointer Rectengle
        matrix_paper = set_main_contour_pointer_rectengle(main_contour)
        cropper_paper = cv2.warpPerspective(copy, matrix_paper, sizepaper)
        cv2.imshow("Cropped", cropper_paper)
        imageResult = cropper_paper.copy()
        # WRAPPER SMALL RECTANGLES IN PAGE
        cv2.waitKey(1)
        # cv2.imwrite('cropper.jpg', imageResult)
        ##################################################################################################
        imgGray = cv2.cvtColor(imageResult, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.GaussianBlur(imageResult, (7, 7), 0)
        # imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
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
        contours, hierarchy = cv2.findContours(imgThresholdCanny2, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        centroids, imgContours = check_small_rectangles_centroid1(contours, imgContours, area_size)

        # sort the centroids by y coordinate
        if len(centroids) > 0 and len(centroids) == 6:
            centroids = sorted(centroids, key=lambda x: x[1])
            if centroids[2][0] < centroids[3][0]:
                w1 = centroids[2]
                w2 = centroids[3]
            else:
                w1 = centroids[3]
                w2 = centroids[2]
            ycenter = [(((w2[0] - w1[0]) // 2) + w1[0]), w1[1]]
            htopcenter = [ycenter[0], (ycenter[1] - centroids[0][1]) // 2 + centroids[0][1] - 15]
            hbelowcenter = [ycenter[0], (centroids[-1][1] - ycenter[1]) // 2 + ycenter[1] + 70]
            tophleft = [w1[0] + 45, htopcenter[1]]
            belowleft = [w1[0] + 45, hbelowcenter[1]]
            pstC1 = np.float32([tophleft, htopcenter, belowleft, hbelowcenter])
            pstC2 = np.float32([[0, 0], [175, 0], [0, 320], [175, 320]])

            # write color
            cv2.circle(imgContours, w1, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, 'w1', w1[0], w1[1])
            cv2.circle(imgContours, w2, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "w2", w2[0], w2[1])
            cv2.circle(imgContours, ycenter, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "ycenter", ycenter[0], ycenter[1])
            cv2.circle(imgContours, htopcenter, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "htop", htopcenter[0], htopcenter[1])
            cv2.circle(imgContours, hbelowcenter, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "hbelow", hbelowcenter[0], hbelowcenter[1])
            cv2.circle(imgContours, tophleft, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "tophleft", tophleft[0], tophleft[1])
            cv2.circle(imgContours, belowleft, 3, (255, 255, 0), -1)
            imgContours = puttext_on_image1(imgContours, "belowleft", belowleft[0], belowleft[1])

            # checkimg = ut.four_point_transform(imgBigContour, wrapPoints)

            matrix1 = cv2.getPerspectiveTransform(pstC1, pstC2)
            croper1 = cv2.warpPerspective(imgBigContour, matrix1, (175, 320))
            gray = cv2.cvtColor(croper1, cv2.COLOR_BGR2GRAY)
            # blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # thresh = cv2.erode(thresh, None, iterations=3)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            CntEdits = []
            centroids1 = []
            for c in cnts:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) > 4:
                    M = cv2.moments(c)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(croper1, (cx, cy), 3, (0, 255, 0), -1)
                    # cv2.drawContours(croper1, [approx], 0, (0, 255, 0),)
                    centroid = (cx, cy)
                    centroids1.append(centroid)
            cv2.imshow("croper", croper1)

            cv2.imshow("thresh", thresh)

        # cv2.imwrite("choine.jpg", imgContours)
        cv2.imshow("Contours", imgContours)
        cv2.waitKey(1)


    img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    cv2.imshow("Main outline", img)
    cv2.waitKey(1)
