import cv2
import os

current_folder = os.listdir('.')



## 이미지 저장
img = cv2.imread('./identicard.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('img_save.jpg', img)
print(result) # 저장이 되었으면 True값 반환


## 동영상 저장
cap = cv2.VideoCapture('./lecture/assets/cat_video.mp4')

# 코덱 정의
FOURCC = cv2.VideoWriter_fourcc(*'DIVX')
WIDTH = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) # FPS를 조절하여 프레임당 시간을 조절할 수 있다
out = cv2.VideoWriter('output.avi', FOURCC, FPS, (WIDTH, HEIGHT))

while cap.isOpened():
  ret, frame = cap.read()
  
  if not ret:
    break
  
  out.write(frame) # 영상 데이터만 저장 (소리 X)
  cv2.imshow('video', frame)
  if cv2.waitKey(25) == ord('q'):
    break
  
  
  
out.release() # 자원 해제
cap.release()
cv2.destroyAllWindows()