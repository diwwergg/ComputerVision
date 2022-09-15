import numpy as np
import matplotlib.pyplot as plt
import cv2

A = plt.imread('mri.png')[:, :, 0]
B = []
for i in range(8):
    B.append((A // 2**i % 2) * 255)
    cv2.imshow('Bit ' + str(i), B[-1])


cv2.waitKey()
