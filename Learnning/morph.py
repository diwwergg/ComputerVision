import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

A = np.zeros((500, 500))
A[200:300,100:200] = 1
A[200:300,250:350] = 1
A[350:450,250:350] = 1
B = measure.label(A)
print(B.max())
plt.imshow(B == 3)
plt.show()