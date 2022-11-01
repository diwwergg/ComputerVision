import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology

def show_image_list(imglist, titlelist, rows:int, cols:int):
    fig = plt.figure()
    for i in range(len(imglist)):
        fig.add_subplot(rows, cols, i+1)
        plt.title(titlelist[i])
        if len(imglist[i].shape) == 2:
            plt.imshow(imglist[i], cmap='gray')
        else:
            plt.imshow(imglist[i])
    plt.show()