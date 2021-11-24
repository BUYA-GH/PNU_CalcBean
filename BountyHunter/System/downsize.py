import cv2
import matplotlib.pyplot as plt
import numpy as np

kong_bgr = cv2.imread('../../Open/t10/5.jpg')
kong_bgr = cv2.resize(kong_bgr, (1024, 768))

kong_rgb = cv2.cvtColor(kong_bgr, cv2.COLOR_BGR2RGB)

# 사각형 좌표: 시작점의 x,y  ,넢이, 너비
rectangle = (180, 10, 900, 750)

mask = np.zeros(kong_rgb.shape[:2], np.uint8)
# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(kong_rgb, # 원본 이미지
           mask,       # 마스크
           rectangle,  # 사각형
           bgdModel,   # 배경을 위한 임시 배열
           fgdModel,   # 전경을 위한 임시 배열 
           5,          # 반복 횟수
           cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화
# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱행 배경을 제외
image_rgb_nobg = kong_rgb * mask_2[:, :, np.newaxis]

# plot
plt.imshow(image_rgb_nobg)
plt.show()