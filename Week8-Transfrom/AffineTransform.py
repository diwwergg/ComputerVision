import numpy as np
import matplotlib.pyplot as plt
def display(obj, color='b'):
    temp = obj.copy()
    temp = np.vstack((temp, temp[0]))
    plt.plot(temp[:, 0], temp[:, 1], color)
def transform(obj, T):
    temp = obj.copy()
    temp = np.hstack((temp, np.ones((len(obj), 1))))
    temp = T @ temp.T
    return temp.T[:, :-1]
obj = np.array([[5, 6], [5, 3], [10, 3], [10, 6]])
display(obj)
# Translation
T1 = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]])
obj_T1 = transform(obj, T1)
display(obj_T1, 'r')
# Rotation
degree = 30; rad = degree * np.pi / 180
T2 = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
obj_T2 = transform(obj, T2)
display(obj_T2, 'g')
# Scaling
scale = 1.5
T3 = np.array([[scale, 0, -3], [0, scale, 0], [0, 0, 1]])
obj_T3 = transform(obj, T3)
display(obj_T3, 'y')
plt.xlim(0, 15); plt.ylim(0, 15); plt.axis('equal'); plt.show()