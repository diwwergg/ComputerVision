import numpy as np
import matplotlib.pyplot as plt
import cv2

A = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    A[:, i] = [i, 0, 0]

print(A)
# plt.imshow(A, cmap='gray')
plt.imshow(A)
plt.show()

# cv2.imshow('A', A.astype(np.uint8))
# cv2.waitKey()
