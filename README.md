### OpenCV 소개

## 컴퓨터 비전을 위한 OpenCV

- 영상 처리와 컴퓨터 비전을 위한 오픈소스 라이브러리
- C, C++, Python 등에서 사용 가능

# cv2.imread(file_name, flag)

- 이미지를 읽어 Numpy 객체로 만드는 함수
- file_name : 읽고자 하는 이미지 파일
- flag : 이미지를 읽는 방법 설정
  IMREAD_COLOR : 이미지를 Color로 읽고 투명한 부분은 무시
  IMREAD_GRAYSCALE : 이미지를 Gracysclae로 읽기
  IMREAD_UNCHANGED : 이미지를 Color로 읽고, 투명한 부분도 읽기(Alpha)

- 반환 값 : Numpy (행, 열, 색상:BGR)

# cv2.imshow(title, image) 특정한 이미지를 화면에 출력합니다.

- title : 윈도우 창의 제목
- image : 출력할 이미지 객체

# cv2.mwrite(file_name, image) 특정한 이미지를 파일로 저장하는 함수

- file_name : 저장한 이미지 파일 이름
- image : 저장할 이미지 객체

# cv2.waitKey(time - ms) 키보드 입력을 처리하는 함수

- time : 입력 대기 시간 (무한대기: 0)
