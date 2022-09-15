import numpy as np
import matplotlib.pyplot as plt
import cv2

def image2bitplane(Imange):
    image = []
    for c in range(3):
        for i in range(8):
            image.append((Imange[:, :, c] // 2 ** i % 2))
    return image
def bitplane2image(Bitplane):
    Chanel = np.zeros((Bitplane[0].shape[0], Bitplane[0].shape[1], 3), dtype=np.uint8)
    for j in range(len(Bitplane)):
        chanel = j // 8
        Chanel[:, :, chanel] = Chanel[:, :, chanel] + Bitplane[j] * 2**(j % 8)
    return Chanel


cat = plt.imread('cat1080.png')
dog = plt.imread('dog1080.png')
# catVga = (cat * 255).astype(np.uint8)[:, :, ::-1]
# dogVga = (dog * 255).astype(np.uint8)[:, :, ::-1]
catB = image2bitplane(cat)
dogB = image2bitplane(dog)
cat_ = bitplane2image(catB[::2] + dogB[::2])

plt.figure()
plt.imshow(cat)
plt.figure()
plt.imshow(dog)
plt.figure()
plt.imshow(cat_)
plt.show()
