import cv2
import numpy as np


## 이미지 변형(원근)

img = cv2.imread('./identicard.jpg')

WIDTH, HEIGHT = 860, 540

src = np.array([[210, 1250],[2650, 1250], [2730, 2760],[200, 2890] ], dtype=np.float32)
dst = np.array([[0, 0],[WIDTH, 0],[WIDTH, HEIGHT],[0, HEIGHT]], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, matrix, (WIDTH, HEIGHT))

cv2.imshow('img', img)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 860 / 540