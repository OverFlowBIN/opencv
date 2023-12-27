import cv2

## IMSHOW

file_path = './identicard.jpg'
img = cv2.imread(file_path) # 해당 경로 파일 불러오기
cv2.imshow('img', img) # img 라는 이름의 창에 img를 표시
key = cv2.waitKey(0) # 지정된 시간동안(ms) 사용자 키 입력 대기
print(key)
cv2.destroyAllWindows() # 모든 창 닫기

## READ

cv2.IMREAD_COLOR # 컬러 이미지, 투명 영역은 무시(기본값)
cv2.IMREAD_GRAYSCALE # 흑백 이미지
cv2.IMREAD_UNCHANGED # 투명 영역까지 포함

file_path = './identicard.jpg'
img_1 = cv2.imread(file_path, cv2.IMREAD_COLOR) 
img_2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
img_3 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) 
cv2.imshow('img_1', img_1)
cv2.imshow('img_2', img_2)
cv2.imshow('img_3', img_3)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

## SHAPE

file_path = './identicard.jpg'
img_1 = cv2.imread(file_path, cv2.IMREAD_COLOR) 
print(img_1.shape) # (4032, 3024, 3) 세로, 가로, Channel
cv2.imshow('img_1', img_1)

key = cv2.waitKey(0)
cv2.destroyAllWindows()