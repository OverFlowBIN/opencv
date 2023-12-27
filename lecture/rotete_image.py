import cv2

## 이미지 회전

# 시계 방향 90도 회전
img = cv2.imread('./screen_shot.png')

rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계 방향 90도 회전
rotate_reverse_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 시계 반대 방향 90도 회전
rotate_180 = cv2.rotate(img, cv2.ROTATE_180) # 시계 방향 90도 회전

cv2.imshow('img', img)
cv2.imshow('rotate_90', rotate_90)
cv2.imshow('rotate_reverse_90', rotate_reverse_90)
cv2.imshow('rotate_180', rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()