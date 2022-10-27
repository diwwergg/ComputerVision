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
centroid = np.mean(obj, axis=0)
T1 = np.array([[1, 0, -centroid[0]], [0, 1, -centroid[1]], [0, 0, 1]])
for degree in range(0, 360):
    rad = degree * np.pi / 180
    T2 = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    obj_T3 = transform(obj, np.linalg.inv(T1) @ T2 @ T1)
    plt.cla()
    display(obj_T3, 'y')
    plt.xlim(0, 15); plt.ylim(0, 15); plt.axis('equal'); plt.pause(.1)