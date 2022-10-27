import numpy as np
import matplotlib.pyplot as plt

def display(obj, color='b', closeLine=True ):
    temp = obj.copy()
    if closeLine:
        temp = np.vstack((temp, temp[0]))
    plt.plot(temp[:, 0], temp[:, 1], color)
def transform(obj, T):
    temp = obj.copy()
    temp = np.hstack((temp, np.ones((len(obj), 1))))
    temp = T @ temp.T
    return temp.T[:, :-1]

def showBlock():
    plt.cla()
    display(Base, 'b', False)
    display(BlockL, 'y')
    display(BlockSquare, 'g')
    plt.xlim(0, 30); plt.ylim(0, 40); plt.axis('equal'); plt.pause(.1)

Base = np.array([[0, 30], [0, 0], [20, 0], [20, 30]])
BlockL = np.array([[0, 7.5], [0, 0], [5, 0], [5, 2.5], [2.5, 2.5], [2.5, 7.5]])
BlockSquare = np.array([[0, 5], [0, 0], [5, 0], [5, 5]])

# Set Start Point
TA1 = np.array([[1, 0, 10], [0, 1, 30], [0, 0, 1]])
BlockL = transform(BlockL, TA1)
TB1 = np.array([[1, 0, 10], [0, 1, 40], [0, 0, 1]])
BlockSquare = transform(BlockSquare, TB1)



time = [True] * 20

while True:
    # Start Down Block A
    if time[0]:
        for y in range(10):
            TA2 = np.array([[1, 0, 0], [0, 1, -1], [0, 0, 1]])
            BlockL = transform(BlockL, TA2)
            showBlock()
        time[0] = False

    #   Rotation BlockA
    if time[1]:
        # Set 0, 0
        centroid = BlockL[3]
        TA2 = np.array([[1, 0, -centroid[0]], [0, 1, -centroid[1]], [0, 0, 1]])
        BlockL = transform(BlockL, TA2)
        # Rotation
        degree = 90
        rad = degree * np.pi / 180
        TA2 = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        BlockL = transform(BlockL, TA2)
        # Set to Before
        TA2 = np.array([[1, 0, centroid[0]], [0, 1, centroid[1]], [0, 0, 1]])
        BlockL = transform(BlockL, TA2)

        showBlock()
        time[1] = False
    #  Down Block A
    if time[2]:
        while BlockL[1][1] > 0:
            TA2 = np.array([[1, 0, 0], [0, 1, -0.5], [0, 0, 1]])
            BlockL = transform(BlockL, TA2)
            showBlock()
        time[2] = False

    # Down BlockSquare
    if time[3]:
        while BlockSquare[1][1] > BlockL[-1][1]:
            TB2 = np.array([[1, 0, 0], [0, 1, -.5], [0, 0, 1]])
            BlockSquare = transform(BlockSquare, TB2)
            showBlock()
        time[3] = False
    showBlock()

