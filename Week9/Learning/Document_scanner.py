import cv2
import numpy as np
from skimage.filters import threshold_local 

def projected_image_size(pts):
    (tl,tr,br,bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  #Below width
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  #Top width
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))   #Right Height
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))   #Left Height
    maxHeight = max(int(heightA), int(heightB))

    dst = np.float32([[0, 0],[maxWidth-1, 0],[0, maxHeight-1],
        [maxWidth-1,maxHeight-1]])

    return (dst,maxWidth,maxHeight)

path = 'Sample_resized.jpg'

def scanner(file_path):
    image = cv2.imread(file_path)  #take input of image
    org = image.copy()
    #cv2.imshow("Orignal",org)
    #cv2.waitkey(0)

    #Resize the image for better and fast processing
    r = 600.0 / image.shape[1]
    dim = (int(image.shape[1]*r),600)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("GrayScaled",image)
    #cv2.waitkey(0)

    image = cv2.bilateralFilter(image, 5, 21, 21) #Blur the image for better edge detection
    #cv2.imshow("Blurred",image)
    #cv2.waitkey(0)

    #Edge detection using canny algorithm
    edged = cv2.Canny(image,30,160)
    #cv2.imshow("Edged",edged)
    #cv2.waitkey(0)

    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    '''Arrange the list of contours found in descending order
    in the basis of contour area'''
    contours.sort(key=cv2.contourArea, reverse=True)

    img = image.copy()
    cv2.drawContours(img,contours,-1,color=(0,0,255),thickness = 3)
    #cv2.imshow("Contoured image",img)
    #cv2.waitkey(0)

    '''Find the main contour that is our main document'''
    for c in contours:
        arc_length = 0.1*cv2.arcLength(c,True)  #Find the perimeter of closed curve
        no_of_points_found = cv2.approxPolyDP(c,arc_length,True)

        if len(no_of_points_found) == 4:
            Main_contour = no_of_points_found
            break
    copy = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img,[Main_contour],-1,(0,255,0),2)
    #cv2.imshow("Main outline",img)
    #cv2.waitkey(0)

    #Get those four points and crop the image
    tl = Main_contour[0][0]
    tr = Main_contour[3][0]
    bl = Main_contour[1][0]
    br = Main_contour[2][0]

    pts = (tl,tr,bl,br)
    pts1 = np.float32([tl,tr,bl,br])

    (pts2,maxWidth,maxHeight) = projected_image_size(pts1)

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    cropped = cv2.warpPerspective(image, matrix,(maxWidth,maxHeight))
    #cv2.imshow("Cropped",cropped)
    #cv2.waitkey(0)

    #For that scanned black and white look
    T = threshold_local(cropped, 13, offset = 10, method = "gaussian")
    cropped = (cropped > T).astype("uint8") * 255
    #cv2.imshow("Scanned",cropped)
    #cv2.waitkey(0)

    #print("Image Scanned Successfully")
    #cv2.destroyAllWindows()
    return cropped