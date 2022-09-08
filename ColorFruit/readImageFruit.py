import numpy as np
import matplotlib.pyplot as plt
import cv2

A = plt.imread('fruit.png')
B = A.copy()
bg = A[:, :, 3] == 0
B[bg] = [0, 0, 0, 1]
C = B[:, :, :3]
r = C[:, :, 0]
g = C[:, :, 1]
b = C[:, :, 2]

plt.subplot(2, 2, 1); plt.imshow(C)
plt.subplot(2, 2, 2); plt.imshow(r, cmap='gray')
plt.subplot(2, 2, 3); plt.imshow(g, cmap='gray')
plt.subplot(2, 2, 4); plt.imshow(b, cmap='gray')
resberry = (r > 0.7) & (g < 0.3) & (b < 0.3 )
D = C.copy()
C[np.logical_not(resberry)] = [0, 0, 0]
plt.figure(); plt.imshow(C)

blueberry = (r > 0.1) & (r < 0.4)
D[np.logical_not(blueberry)] = [255, 255, 255]
plt.figure(); plt.imshow(D)

plt.show()
