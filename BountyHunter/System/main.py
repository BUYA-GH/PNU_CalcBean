import cv2
import os
import time

from dataset import dishSegmentation, colorSegmentation, countCircle

# Now only read 5th image
DATA_ADDR = ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10',
            't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20',
            't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30']

IMG_ADDR = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

openImgAddr = '../../Open/'
hiddenImgAddr = '../../Hidden/'

resultFileAddr = '../Out/Kong_BountyHunter.txt'
file = open(resultFileAddr, 'w', encoding='utf8')
file.write('현상금 사냥꾼\n')

now = time.localtime()
file.write('%02d-%02d-%02d-%02d-%02d\n'%(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))

imageSize = (800, 600)
calcResult = []

start = time.time()
for i in DATA_ADDR:
    kong_bgr = cv2.imread(os.path.join( openImgAddr, i, IMG_ADDR[4] ))
    #kong_bgr = cv2.imread( openImgAddr+DATA_ADDR[i] )
    kong_bgr = cv2.medianBlur(kong_bgr, 3)
    kong_bgr = cv2.resize(kong_bgr, imageSize)

    dishImage = dishSegmentation(kong_bgr, imageSize)
    kongImage = colorSegmentation(dishImage)
    calcResult.append(countCircle(kongImage))
end = time.time()

file.write('%ds\n'%(end-start))
file.write('%d\n'%(len(DATA_ADDR)))
for i in range(len(DATA_ADDR)):
    file.write('T%02d\t%d\n'%(i+1, calcResult[i]))


file.close()