import cv2


img = cv2.imread('./screen_shot.png')


## 고정 크기로 설정

# dst = cv2.resize(img, (200, 300)) # width, height

# cv2.imshow('dst', dst)
# cv2.waitKey(0)



## 비율로 크기 설정

dst = cv2.resize(img, None, fx = 2, fy = 2) # x, y 비율로 정의

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)


## 보간법
"""
cv2.INTER_AREA : 크기를 줄일 때 사용
cv2.INTER_CUBIC : 크기를 늘릴 때 사용 (속도 느림, 퀄리티 좋음)
cv2.INTER_LINEAR : 크기를 늘릴 때 사용 (기본값)
"""

dst = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)

cv2.imshow('img', img) 
cv2.imshow('dst2', dst)
cv2.waitKey(0)


cap = cv2.VideoCapture('./lecture/assets/cat_video.mp4')

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  frame_resize = cv2.resize(frame, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
  
  cv2.imshow('video', frame_resize)
  if cv2.waitKey(25) == ord('q'):
    break
  
cap.release()





cv2.destroyAllWindows()






