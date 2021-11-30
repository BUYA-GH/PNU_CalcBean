import cv2
import matplotlib.pyplot as plt
import numpy as np

def dishSegmentation(img_bgr, imgSize):
    # 사각형 좌표: 시작점의 x,y  ,넢이, 너비
    rectangle = (int(imgSize[0]*1/8), 0, int(imgSize[0]*7/8), imgSize[1])

    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    # grabCut에 사용할 임시 배열 생성
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # grabCut 실행
    cv2.grabCut(img_bgr, # 원본 이미지
           mask,       # 마스크
           rectangle,  # 사각형
           bgdModel,   # 배경을 위한 임시 배열
           fgdModel,   # 전경을 위한 임시 배열 
           5,          # 반복 횟수
           cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화
    # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # 이미지에 새로운 마스크를 곱행 배경을 제외
    image_bgr_nobg = img_bgr * mask_2[:, :, np.newaxis]

    return image_bgr_nobg

def colorSegmentation(dish):
    dish = cv2.cvtColor(dish, cv2.COLOR_BGR2RGB)
    hsv_dish = cv2.cvtColor(dish, cv2.COLOR_RGB2HSV)

    low = np.array([0, 60, 0])
    high = np.array([55, 190, 255])

    mask = cv2.inRange(hsv_dish, low, high)
    res = cv2.bitwise_and(dish, dish, mask=mask)

    return res