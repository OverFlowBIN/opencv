import cv2

cap = cv2.VideoCapture('./lecture/cat_video.mp4')

while cap.isOpened():
  ret, frame = cap.read() # ret: 성공여부, frame: 받아온 이미지(프레임)
  if not ret:
    print('더 이상 가져올 프레임이 없다')
    break 
    
  cv2.imshow('video', frame)
  if cv2.waitKey(1) == ord('q'): # waitKey의 시간(ms)을 조절하여 영상 속도를 조절할 수 있다
    print('사용자 입력에 의해 비디오 종료')
    break
  
cap.release()
cv2.destroyAllWindows()