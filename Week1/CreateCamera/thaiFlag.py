import numpy as np
import matplotlib.pyplot as plt
import cv2

R = np.zeros((100, 900, 3), dtype=np.uint8)
R[:, :] = [239, 51, 64]
W = np.zeros((100, 900, 3), dtype=np.uint8)
W[:, :] = [255, 255, 255]
B = np.zeros((200, 900, 3), dtype=np.uint8)
B[:, :] = [0, 36, 125]

Thai = np.vstack((R, W, B, W, R))

plt.imshow(Thai)
plt.axis(False)
plt.show()

# cv2.imshow('A', A.astype(np.uint8))
# cv2.waitKey()
