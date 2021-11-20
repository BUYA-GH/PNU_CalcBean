import cv2
import matplotlib.pyplot as plt
import numpy as np

bird = cv2.imread('../../Open/t01/5.jpg')

plt.imshow(bird)
plt.show()

hsv_bird = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)

low_blue = np.array([0, 55, 0])
high_blue = np.array([255, 118, 255])
mask = cv2.inRange(hsv_bird, low_blue, high_blue)

plt.imshow(mask)
plt.show()

res = cv2.bitwise_and(bird, bird, mask=mask)

plt.imshow(res)
plt.show()