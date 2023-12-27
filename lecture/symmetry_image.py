import cv2

## 이미지 대칭


img = cv2.imread('./screen_shot.png')

# 좌우 대칭
flip_horizontal = cv2.flip(img, 1) # flipCode > 0 : 좌우 대칭(Horizontal)
flip_vertial = cv2.flip(img, 0) # flipCode == 0 : 상하 대칭(vertical)
flip_both = cv2.flip(img, -1) # flipCode == 0 : 상하 대칭(vertical)

cv2.imshow('img', img)
cv2.imshow('flip_horizontal', flip_horizontal)
cv2.imshow('flip_vertial', flip_vertial)
cv2.imshow('flip_both', flip_both)

cv2.waitKey(0)
cv2.destroyAllWindows()