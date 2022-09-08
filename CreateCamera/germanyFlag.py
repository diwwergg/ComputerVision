import numpy as np
import matplotlib.pyplot as plt
import cv2


def germany():
    Red = np.zeros((100, 500, 3), dtype=np.uint8)
    Red[:, :] = [211, 0, 0]
    Yellow = np.zeros((100, 500, 3), dtype=np.uint8)
    Yellow[:, :] = [255, 206, 0]
    Black = np.zeros((100, 500, 3), dtype=np.uint8)
    Black[:, :] = [0, 0, 0]
    Germany = np.vstack((Black, Red, Yellow))
    return Germany


def france():
    # 450, 750
    B = np.zeros((250, 150, 3), dtype=np.uint8)
    B[:, :] = [0, 85, 164]
    W = np.zeros((250, 150, 3), dtype=np.uint8)
    W[:, :] = [255, 255, 255]
    R = np.zeros((250, 150, 3), dtype=np.uint8)
    R[:, :] = [239, 65, 53]
    France = np.hstack((B, W, R))
    return France

def italy():
    G= np.zeros((200, 100, 3), dtype=np.uint8)
    G[:, :] = [0, 135, 99]
    W=np.zeros((200, 100, 3), dtype=np.uint8)
    W[:,:] = [240, 238, 233]
    R=np.zeros((200, 100, 3), dtype=np.uint8)
    R[:, :] = [206, 41, 57]
    Italy = np.hstack((G, W, R))
    return Italy

# def japan():
#     Japan = np.full((400, 600, 3), 255, np.uint8)
#     for x in range(Japan.shape[1]):
#         for y in range(Japan.shape[0]):
#             if (x-50)
plt.figure()
plt.imshow(germany())

plt.figure()
plt.imshow(france())
# plt.axis(False)

plt.figure()
plt.imshow(italy())
plt.show()



# cv2.imshow('A', A.astype(np.uint8))
# cv2.waitKey()
