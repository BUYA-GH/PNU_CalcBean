import cv2
import matplotlib.pyplot as plt
import numpy as np

kong = cv2.imread('../../Open/t01/5.jpg')
kong = cv2.cvtColor(kong, cv2.COLOR_BGR2RGB)

hsv_kong = cv2.cvtColor(kong, cv2.COLOR_RGB2HSV)
low = np.array([0, 60, 0])
high = np.array([55, 190, 255])

mask = cv2.inRange(hsv_kong, low, high)
res = cv2.bitwise_and(kong, kong, mask=mask)

plt.imshow(res)
plt.show()