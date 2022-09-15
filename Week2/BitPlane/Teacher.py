import numpy as np
import matplotlib.pyplot as plt
import cv2
def im2bp(A):
    B = []
    for c in range(3):
        for i in range(8):
            B.append(A[:, :, c] // 2**i % 2)
    return B
def bp2im(B):
    C = np.zeros((B[0].shape[0], B[0].shape[1], 3), dtype=np.uint8)
    for j in range(len(B)):
        c = j // 8
        C[:, :, c] = C[:, :, c] + B[j] * 2**(j % 8)
    return C
a = plt.imread('cat1080.png')
b = plt.imread('dog1080.png')
a_bp = im2bp(a)
b_bp = im2bp(b)
a_ = bp2im(a_bp[::2] + b_bp[::2])
plt.imshow(a_)
plt.show()