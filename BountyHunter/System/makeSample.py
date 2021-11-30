import cv2
import os

from dataset import colorSegmentation, dishSegmentation


# Now only read 5th image
DATA_ADDR = ['t08', 't09', 't12', 't07', 't30', 
            't27', 't17', 't29', 't25', 't28', 't13']
IMG_ADDR = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
openImgAddr = '../../Open/'
sampleImgAddr = 'Sample'

imageSize = (800, 600)

for i in DATA_ADDR:
    for j in IMG_ADDR:
        kong_bgr = cv2.imread(os.path.join(openImgAddr, i, j))
        kong_bgr = cv2.resize(kong_bgr, imageSize)

        kong_bgr = dishSegmentation(kong_bgr, dishSegmentation)
        kongImage = colorSegmentation(kong_bgr)
        cv2.imwrite(os.path.join(sampleImgAddr, i, j), kongImage)