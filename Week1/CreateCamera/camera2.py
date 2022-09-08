import numpy as np
import matplotlib.pyplot as plt
import cv2

Z = [[0, 0, 0], [1, 1, 1]]
A = np.array(Z)
B = np.vstack((np.zeros((20, 100)), np.ones((20, 100))))
print(Z)
print(A)

# Matplotlib show
# plt.imshow(A, cmap='gray')
# plt.show()

# OpenCv Show
cv2.imshow('B', (B * 255).astype(np.uint8))
cv2.waitKey()
