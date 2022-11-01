from email.mime import image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology

myAnswer = [3, 2, 1, 1, 4,
            3, 2, 3, 1, 2]

# SET PAPER
# pixel a4 paper size
size = (width, height) = (825, 1170)
# read image
img = cv2.imread('papersheet4.jpg')
# img resize 540 x 960
img = cv2.resize(img, (540, 960))
img2= img.copy()
# onclick select paper 4 point
def select_point(event, x, y, flags, param):
    global point, point_count
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img2, (x, y), 5, (0, 0, 255), -1)
        point.append([x, y])
        point_count += 1
# show Original image
cv2.namedWindow('image')
cv2.setMouseCallback('image', select_point)
point = []
point_count = 0
while True:
    cv2.imshow('image', img2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if point_count == 4:
        break
cv2.destroyAllWindows()
# get 4 point
pts1 = np.float32(point)
# get 4 point
pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
# get matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)
# get image
result = cv2.warpPerspective(img, matrix, size)
plt.imshow(result)
plt.show()
# ==================================================================================================

# DETECT RECTANGLE CONNER
image = result.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray < 90



plt.figure('gray')
plt.imshow(gray, cmap='gray')

plt.show()

