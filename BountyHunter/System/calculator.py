import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import dishSegmentation, colorSegmentation

#DATA_ADDR = ['t08', 't09', 't12', 't07', 't30', 
#            't27', 't17', 't29', 't28', 't13']
DATA_ADDR = ['0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9']
DATA_ANS = [196, 263, 379, 513, 691,
            1032, 1190, 1323, 1429, 1600]

IMG_ADDR = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
sampleImgAddr = 'Sample'

def countCircle(kong):
    kong_gray = cv2.cvtColor(kong, cv2.COLOR_RGB2GRAY)
    blr = cv2.GaussianBlur(kong_gray, (0, 0), 1)
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 18, param1=50, param2=15, minRadius=8, maxRadius=13)

    if circles is not None:
        return circles.shape[1]
    else:
        return 0

def ensembleCount(addr, imageSize):
    addrImages = []
    sampleImages = []
    ansArray = []

    for i in IMG_ADDR :
        kong = cv2.imread(os.path.join( addr, i ))
        kong = cv2.resize(kong, imageSize)
        kong = colorSegmentation(kong)

        kong = cv2.cvtColor(kong, cv2.COLOR_RGB2HSV)
        kong = cv2.calcHist([kong], [0, 1], None, [55,190], [0,55,60,190])
        #cv2.normalize(kong, kong, 0, 1, cv2.NORM_MINMAX)

        addrImages.append(kong)

    
    for i in DATA_ADDR :
        arr = []
        for j in IMG_ADDR : 
            oppe = cv2.imread(os.path.join(sampleImgAddr, i, j))
            oppe = cv2.cvtColor(oppe, cv2.COLOR_BGR2HSV)
            oppe = cv2.calcHist([oppe], [0, 1], None, [55,190], [0,55,60,190])
            #cv2.normalize(oppe, oppe, 0, 1, cv2.NORM_MINMAX)

            arr.append(oppe)
        sampleImages.append(arr)

    for i in range(len(IMG_ADDR)) :
        kong = addrImages[i]
        temp = []
        for j in range(len(DATA_ADDR)) :
            oppe = sampleImages[j][i]

            ret = cv2.compareHist(kong, oppe, cv2.HISTCMP_CORREL)
            temp.append(ret)
        
        ans = temp.index(max(temp))
        ansArray.append(DATA_ANS[ans])

    ret = np.array(ansArray)
    return ret.mean()