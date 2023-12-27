import cv2
import numpy as np

## 빈 스케치북 그리기

# 세로 480 x 가로 640, Channle 3 (RGB)
img =  np.zeros((480, 640, 3), dtype = np.uint8)
# img[:] = (255, 255, 255) # 전체 공간을 흰색으로 채우기


## 일부 영역 색칠
# img[100:200, 200:300] = (255, 255, 255) # [세로영역, 가로영역]


## 직선 그리기
"""
직선의 종류
cv2.LINE_4: 상하좌우 4방향으로 연결된 선
cv2.LINE_8: 대각선을 포함한 8 방향으로 연결된 선(기본값)
cv2.LINE_AA: 부드러운 선 (anti-aliasing)
"""

COLOR = (0, 255, 255) # BRG
THICKNESS = 3 

# cv2.line 이미지, 시작 점, 끝 점, 색깔, 두께, 선 종류
cv2.line(img, (50,100), (400, 50), COLOR, THICKNESS, cv2.LINE_8)
cv2.line(img, (50,200), (400, 150), COLOR, THICKNESS, cv2.LINE_4)
cv2.line(img, (50,300), (400, 250), COLOR, THICKNESS, cv2.LINE_AA)


## 원 그리기

COLOR = (255, 255, 0)
RADIUS = 50 # 반지름
# cv2.circle 이미지, 원의 중심점, 반지름, 색깔, 두께, 선 종류
cv2.circle(img, (200, 100), RADIUS, COLOR, THICKNESS, cv2.LINE_AA) # 속이 빈 원
cv2.circle(img, (300, 200), RADIUS, COLOR, cv2.FILLED, cv2.LINE_AA) # 속이 빈 원


## 사각형 그리기
COLOR = (0, 255, 0)
# 이미지, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께
cv2.rectangle(img, (300, 300), (200, 200), COLOR, THICKNESS) # 속이 빈 사각형
cv2.rectangle(img, (500, 500), (300, 300), COLOR, cv2.FILLED) # 속이 꽉찬 사각형


## 다갹형 그리기
COLOR = (0, 0, 255)
pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[300, 300], [300, 400], [400, 300], [400, 400]])
pts3 = np.array([[200, 100], [250, 150], [150,200], [200, 200], [300, 300]])
# 이미지, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께
cv2.polylines(img, [pts1], False, COLOR, THICKNESS, cv2.LINE_AA) # 닫히지 않은 폴리선
cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA) # 닫한 폴리선
cv2.fillPoly(img, [pts3], COLOR, cv2.LINE_AA) # 꽉찬 다각형



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()