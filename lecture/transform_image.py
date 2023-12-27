import cv2


## 이미지 변형(흑백)

# img = cv2.imread('./screen_shot.png', cv2.IMREAD_GRAYSCALE) => 이미지를 불러올 때 부터 흑백으로 불러옴
img = cv2.imread('./screen_shot.png')

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


## 이미지 블러 처리(가우시안 블러)
# 커널 사이즈 변화에 따른 흐름 
# [3, 3], [5, 5], [7, 7] 보통 이 셋중에 하나를 적용해서 사용

kernel_3 = cv2.GaussianBlur(img, (3,3), 0)
kernel_5 = cv2.GaussianBlur(img, (5,5), 0)
kernel_7 = cv2.GaussianBlur(img, (7,7), 0)

# 표준 편차 변화에 따른 흐름

sigma_1 = cv2.GaussianBlur(img, (0,0), 1)
sigma_2 = cv2.GaussianBlur(img, (0,0), 2)
sigma_3 = cv2.GaussianBlur(img, (0,0), 3)


cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.imshow('kernel_3', kernel_3)
cv2.imshow('kernel_5', kernel_5)
cv2.imshow('kernel_7', kernel_7)

cv2.imshow('sigma_1', sigma_1)
cv2.imshow('sigma_2', sigma_2)
cv2.imshow('sigma_3', sigma_3)
cv2.waitKey(0)
cv2.destroyAllWindows()