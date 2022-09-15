import numpy as np
import matplotlib.pyplot as plt
import cv2
def imageToBitplane(Imange):
    image = []
    for c in range(3):  # 0 = R, G = 1, B = 2
        for i in range(8):   # 8 bit 0 - 7
            image.append(Imange[:, :, c] // 2 ** i % 2)
    return image
def bitplaneToImage(Bitplane):
    Channel = np.zeros((Bitplane[0].shape[0], Bitplane[0].shape[1], 3), dtype=np.uint8)
    for j in range(0, len(Bitplane)):
        channel = (j) // 8  # answer channel is 0, 1, 2
        Channel[:, :, channel] = Channel[:, :, channel] + Bitplane[j] * 2**(j % 8)
    return Channel

# Mission 1
cat = plt.imread('cat1080.png')
dog = plt.imread('dog1080.png')
catInt255 = (cat*255).astype(np.uint8)
dogInt255 = (dog * 255).astype(np.uint8)
catB = imageToBitplane(catInt255)
dogB = imageToBitplane(dogInt255)
plt.figure('cat')
plt.imshow(cat)
plt.figure('dog')
plt.imshow(dog)

# Mission 2 A = Cat , B = Dog
dogBitLess = []
catBitMore = []
for k in range(24):
    if( k % 8 < 4):
        dogBitLess.append(dogB[k])
    else:
        catBitMore.append(catB[k])

catSumDog = bitplaneToImage(catBitMore + dogBitLess)
plt.figure('catSumDog')
plt.imshow(catSumDog)

# Mission 3
catSumDogBp = imageToBitplane(catSumDog)
dogBitMore = []
catDogBitLess = []
for l in range(24):
    if( l % 8 < 4):
        catDogBitLess.append(catSumDogBp[l])
    else:
        dogBitMore.append(dogB[l])
BigDogSmallCatDogDak = bitplaneToImage(dogBitMore + catDogBitLess)
plt.figure('BigDogSmallCatDogDak')
plt.imshow(BigDogSmallCatDogDak)
plt.show()
