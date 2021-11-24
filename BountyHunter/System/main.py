import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset import dishSegmentation, colorSegmentation

# Now only read 5th image
DATA_ADDR = ['t01/5.jpg', 't02/5.jpg', 't03/5.jpg', 't04/5.jpg', 't05/5.jpg', 't06/5.jpg', 't07/5.jpg', 't08/5.jpg', 't09/5.jpg', 't10/5.jpg',
            't11/5.jpg', 't12/5.jpg', 't13/5.jpg', 't14/5.jpg', 't15/5.jpg', 't16/5.jpg', 't17/5.jpg', 't18/5.jpg', 't19/5.jpg', 't20/5.jpg',
            't21/5.jpg', 't22/5.jpg', 't23/5.jpg', 't24/5.jpg', 't25/5.jpg', 't26/5.jpg', 't27/5.jpg', 't28/5.jpg', 't29/5.jpg', 't30/5.jpg']

openImgAddr = '../../Open/'
hiddenImgAddr = '../../Hidden/'

imageSize = (1024, 768)

#for i in range(len(DATA_ADDR)):
kong_bgr = cv2.imread( openImgAddr+DATA_ADDR[7] )
kong_bgr = cv2.resize(kong_bgr, imageSize)

dishImage = dishSegmentation(kong_bgr)
kongImage = colorSegmentation(dishImage)

plt.imshow(kongImage)
plt.show()