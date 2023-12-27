import cv2


## 이미지 자르기
# 영역을 잘라서 새로운 윈도우에 표시

img = cv2.imread('./screen_shot.png')
shape = img.shape
print(shape)


crop = img[100:200, 200:400] # 세로 범위, 가로범위 (사각형으로 짤림)



# 영역을 잘라서 기존이미지에 표시
img[100:200, 400:600] = crop

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()