import cv2

cap = cv2.VideoCapture(0) # 0번째 카메라 장치(Device ID) 카메라가 2개 있으면 0, 1중 선택하여 사용

if not cap.isOpened(): # 카메라가 잘 열리지 않은 경우
  exit() # 프로그램 종류
  
while True:
  ret, frame = cap.read()
  if not ret:
    break
  
  cv2.imshow('camera', frame)
  if cv2.waitKey(1) == ord('q'): # 사용자가 q 를 입력하면
    break

cap.release()
cv2.destroyAllWindows()