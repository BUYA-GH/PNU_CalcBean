import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

kong = cv2.imread('../../Open/t01/5.jpg')
kong = cv2.cvtColor(kong, cv2.COLOR_BGR2RGB)


hsv_kong = cv2.cvtColor(kong, cv2.COLOR_RGB2HSV)
low_blue = np.array([0, 60, 0])
high_blue = np.array([55, 190, 255])

lo_square = np.full((10, 10, 3), low_blue, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), high_blue, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

mask = cv2.inRange(hsv_kong, low_blue, high_blue)

plt.imshow(mask)
plt.show()

res = cv2.bitwise_and(kong, kong, mask=mask)

plt.imshow(res)
plt.show()