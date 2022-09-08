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

D = C.copy()
E = C.copy()
F = C.copy()

plt.subplot(2, 2, 1); plt.imshow(C)
plt.subplot(2, 2, 2); plt.imshow(r, cmap='gray')
plt.subplot(2, 2, 3); plt.imshow(g, cmap='gray')
plt.subplot(2, 2, 4); plt.imshow(b, cmap='gray')
redberry = (r > 0.7) & (g < 0.3) & (b < 0.3)
D[np.logical_not(redberry)] = [0, 0, 0]
plt.figure('Redberry'); plt.imshow(D)

blueberry = (r > 0.1) & (r < 0.4) & (g > 0.15) & (g < 0.36) & (b > 0.3)
E[np.logical_not(blueberry)] = [255, 255, 255]
plt.figure('Blueberry'); plt.imshow(E)

banana = (r > 0.8) & (g > 0.6) & (g < 0.88) & (b > 0.2) & (b < 0.6)
F[np.logical_not(banana)] = [0, 0, 0]
plt.figure('Banana'); plt.imshow(F)
plt.show()
plt.waitforbuttonpress()
