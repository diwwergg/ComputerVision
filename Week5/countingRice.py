import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology

A = plt.imread('rice.tif')
plt.figure('A')
plt.imshow(A); plt.show()

B1 = A[:150, :] > 145
B2 = A[150:, :] > 110
# plt.figure('B1'); plt.imshow(B1)
# plt.figure('B2'); plt.imshow(B2)
# plt.show()

B1 = morphology.erosion(B1, np.ones((3, 3)))
B2 = morphology.erosion(B2, np.ones((3, 3)))
# plt.figure('B1'); plt.imshow(B1)
# plt.figure('B2'); plt.imshow(B2)
# plt.show()

B = np.vstack((B1, B2))
# plt.figure('B'); plt.imshow(B)
# plt.show()

B = morphology.erosion(B, np.ones((3, 3)))
B = measure.label(B)
plt.figure('A');plt.imshow(A)
plt.figure('B');plt.imshow(B)
counter = 0
for r in measure.regionprops(B):
    if r.area > 0:
        plt.plot(r.centroid[1], r.centroid[0], 'ro')
        counter += 1
plt.show()
print(counter)


