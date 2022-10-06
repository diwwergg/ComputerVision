import matplotlib.pyplot as plt
import numpy as np
import numpy as num
from skimage import measure, morphology

A = plt.imread('dowels.tif')
plt.imshow(A); plt.show() ;  # 1

B = A > 120
plt.imshow(B); plt.show() ;  # 2

B = morphology.erosion(B, np.ones((8, 8)))
plt.imshow(B); plt.show() ;

B = morphology.dilation(B, np.ones((3, 3)))
plt.imshow(B); plt.show() ;

B = morphology.erosion(B, np.ones((5, 5)))
plt.imshow(B); plt.show() ;

B = measure.label(B)

plt.imshow(B)
counter = 0
for r in measure.regionprops(B):
    if r.area > 500:
        plt.plot(r.centroid[1], r.centroid[0], 'ro')
        counter += 1
plt.show()
