import matplotlib.pyplot as plt
from skimage import morphology, measure
import numpy as np

A = plt.imread('dowels.tif')
plt.imshow(A); plt.show()
B = A > 100
plt.imshow(B); plt.show()
B = morphology.erosion(B, np.ones((8, 8)))
plt.imshow(B); plt.show()
B = morphology.dilation(B, np.ones((3, 3)))
plt.imshow(B); plt.show()
B = morphology.erosion(B, np.ones((5, 5)))
plt.imshow(B); plt.show()
B = measure.label(B)
plt.imshow(B); plt.show()

plt.imshow(B)
counter = 0
for r in measure.regionprops(B):
    if r.area > 150:
        plt.plot(r.centroid[1], r.centroid[0], 'ro')
        counter += 1
plt.show()
print(counter)