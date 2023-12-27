import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def myPutText(src, text, pos, font_size, font_color):
  img_pil = Image.fromarray(src)
  draw = ImageDraw.Draw(img_pil)
  font = ImageFont.truetype('./lecture/gulim.ttf', font_size)
  draw.text(pos, text, font = font, fill = font_color)
  return np.array(img_pil)
  



## OpenCV에서 사용하는 글꼴 종류
"""
cv.FONT_HERSHEY_SIMPLEX : 보통 크기의 산 세리프 글꼴
cv2.FONT_HERSHEY_PLAIN : 작은 크기의 산 세리프 글꼴
cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 스타일 글꼴
cv2.FONT_HERSHEY_TRIPLEX : 보통 크기의 세리프 글꼴
cv2.FONT_ITALIC : 기울임(이텔릭체)
"""

img = np.zeros((600, 800, 3), dtype = np.uint8)

COLOR = (255, 255, 255)
THICKNESS = 1
SCALE = 1

# 이미지, 텍스트 내용, 시작 위치(폰트의 왼쪽 아래), 폰트 종류, 크기, 색깔, 두께
cv2.putText(img, "OpenCV Tutorial", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "OpenCV Tutorial", (20, 150), cv2.FONT_HERSHEY_PLAIN, SCALE, COLOR, THICKNESS)
cv2.putText(img, "OpenCV Tutorial", (20, 250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "OpenCV Tutorial", (20, 350), cv2.FONT_HERSHEY_TRIPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "OpenCV Tutorial", (20, 450), cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, SCALE, COLOR, THICKNESS)


FONT_SIZE = 30
img = myPutText(img, '오픈씨브이 튜토리얼', (20, 550), FONT_SIZE, COLOR)



cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()