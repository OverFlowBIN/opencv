import cv2
import numpy as np
import pytesseract
from PIL import Image
from random import random

def rotate_image(image, angle):
    # 이미지를 지정된 각도로 회전합니다.
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def resize_image(image, width, height):
    # 이미지 크기를 지정된 폭과 높이로 조정합니다.
    resized_image = cv2.resize(image, (width, height))
    return resized_image


# 실제 이미지의 경로로 'your_image_path.jpg'를 대체하세요.
image_path = './identicard.jpg'
# image_path = './test.png'
image = cv2.imread(image_path)


def find_and_mask_numbers(image):

    resized_image = resize_image(image, 1200, 900)


    # 이미지를 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)
    cv2.imshow('gray', gray)


    # 노이즈를 감소시키고 윤곽을 감지하기 위해 가우시안 블러를 적용합니다.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('blurred', blurred)

    # Canny를 사용하여 가장자리를 검출합니다.
    edges = cv2.Canny(blurred, 75, 200, True)
    cv2.imshow('edges', edges)


    # 가장자리 감지된 이미지에서 윤곽을 찾습니다.
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    checkpnt  = 0

    for i in cnts:
        peri = cv2.arcLength(i, True)  # contour가 그리는 길이 반환
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 길이에 2% 정도 오차를 둔다

        if len(approx) == 4:  # 도형을 근사해서 외곽의 꼭짓점이 4개라면 명암의 외곽으로 설정
          screenCnt = approx
          size = len(screenCnt)
          break

        if len(approx) != 4 and checkpnt == 0:  # 사각형이 그려지지 않는다면 grab_cut 실행
            size = 0
            checkpnt += 1
            # grab_cut(resize_img)

        if len(approx) != 4 and checkpnt > 0:
            size = 0

        if (size > 0):
          # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
          cv2.line(resized_image, tuple(screenCnt[0][0]), tuple(screenCnt[size-1][0]), (255, 0, 0), 3)
          for j in range(size-1):
            color = list(np.random.random(size=3) * 255)
            cv2.line(resized_image, tuple(screenCnt[j][0]), tuple(screenCnt[j+1][0]), color, 3)

          #for i in screenCnt: #이렇게 하면 네 변을 다른 색으로 표현 불가능(네 변이 모두 다 똑같은 색으로 나온다.)
              #color = list(np.random.random(size=3) * 256)
              #cv2.drawContours(resized_image, [screenCnt], -1,color, 3)

          # 2.B 추출된 선분(좌, 우, 상, 하)의 기울기, y절편, 양끝점의 좌표를 각각 출력
          axis = np.zeros(4)

          # 기울기 = (y증가량) / (x증가량)
          #left_axis = (screenCnt[0][0][1] - screenCnt[1][0][1]) / (screenCnt[0][0][0] - screenCnt[1][0][0])
          #down_axis = (screenCnt[1][0][1] - screenCnt[2][0][1]) / (screenCnt[1][0][0] - screenCnt[2][0][0])
          #right_axis = (screenCnt[2][0][1] - screenCnt[3][0][1]) / (screenCnt[2][0][0] - screenCnt[3][0][0])
          #upper_axis = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])

          axis[3] = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])
          for k in range(3):
              axis[k] = (screenCnt[k][0][1] - screenCnt[k+1][0][1]) / (screenCnt[k][0][0] - screenCnt[k+1][0][0])

          left_axis = axis[0] #좌 기울기
          down_axis = axis[1] #하 기울기
          right_axis = axis[2] #우 기울기
          upper_axis = axis[3] #상 기울기

          print("(2.B) 순서대로 좌, 우, 상, 하 선분의 기울기")
          print(left_axis, right_axis, upper_axis, down_axis)
          print("\n")

          # y = ax + b 에서 x = 0일때의 b가 y절편 / 기울기를 알고 두 좌표를 알 때의 방정식 : y - y1 = (y2 - y1)/(x2 - x1) * (x - x1)
          #좌 선분의 y절편
          #left_y - screenCnt[1][0][1] = left_axis * (left_x - screenCnt[1][0][0])
          #left_y = (left_axis * left_x) - (left_axis * screenCnt[1][0][0]) + screenCnt[1][0][1]
          #따라서 left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0])
          left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0]) #좌 y절편

          #우 선분의 y절편
          #right_y - screenCnt[3][0][1] = right_axis * (right_x - screenCnt[3][0][0])
          #right_y = (right_axis * right_x) - (right_axis * screenCnt[3][0][0]) + screenCnt[3][0][1]
          #따라서 right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0])
          right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0]) #우 y절편

          #상 선분의 y절편
          #upper_y - screenCnt[0][0][1] = upper_axis * (upper_x - screenCnt[0][0][0])
          #upper_y = (upper_axis * upper_x) - (upper_axis * screenCnt[0][0][0]) + screenCnt[0][0][1]
          #따라서 upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0])
          upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0]) #상 y절편

          #하 선분의 y절편
          #donw_y - screenCnt[2][0][1] = down_axis * (down_x - screenCnt[2][0][0])
          #down_y = (down_axis * down_x) - (down_axis * screenCnt[2][0][0]) + screenCnt[2][0][1]
          #따라서 down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0])
          down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0]) #하 y절편

          print("(2.B) 순서대로 좌, 우, 상, 하 선분의 y절편")
          print(left_y, right_y, upper_y, down_y)
          print("\n")

          #양끝점의 좌표
          print("(2.B) 순서대로 좌, 우, 상, 하 선분의 양 끝점")
          print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1])) #좌 선분의 양 끝점
          print((screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1])) #우 성분의 양 끝점
          print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1])) #상 성분의 양 끝점
          print((screenCnt[1][0][0], screenCnt[1][0][1]), (screenCnt[2][0][0], screenCnt[2][0][1])) #하 성분의 양 끝점
          print("\n")

          # 3.B 네 꼭짓점을 각각 다른 색 점으로 표시한다.
          cv2.drawContours(resized_image, screenCnt, 0, (0, 0, 0), 15) #검
          cv2.drawContours(resized_image, screenCnt, 1, (255, 0, 0), 15) #파
          cv2.drawContours(resized_image, screenCnt, 2, (0, 255, 0), 15) #녹
          cv2.drawContours(resized_image, screenCnt, 3, (0, 0, 255), 15) #적

          cv2.imshow("With_Color_Image", resized_image)


          # 3.C  네 꼭지점(좌상, 좌하, 우상, 우하)의 좌표를 출력한다.
          # vertex = solving_vertex(screenCnt.reshape(4,2))
          #(topLeft, bottomLeft, topRight, bottomRight) = vertex

          print("(3.C) 순서대로 좌상, 좌하, 우상, 우하의 꼭짓점 좌표")
          # print(vertex)

    cv2.waitKey(0)
    cv2.destroyAllWindows()







    # drawContours = cv2.drawContours(resized_image, contours, -1, (0, 0, 225), 3)
    # cv2.imshow('drawContours', drawContours)

    # rect  = cv2.minAreaRect(contours)
    # cv2.imshow('rect', rect)

    # arr = [];

    # for contour in contours:
    #     epsilon = 0.02 * cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, epsilon, True)

    #     if len(approx) == 4:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         arr.append([x, y, w, h])

    #         rectangle = cv2.rectangle(resized_image, (x,y), (x+w, y+h), (255, 0, 0), 3)
    #         cv2.imshow('rectangle', rectangle)



    #         # break


    # print('arr', arr)





# def solving_vertex(pts):
#     points = np.zeros((4,2), dtype= "uint32") #x,y쌍이 4개 쌍이기 때문

#     #원점 (0,0)은 맨 왼쪽 상단에 있으므로, x+y의 값이 제일 작으면 좌상의 꼭짓점 / x+y의 값이 제일 크면 우하의 꼭짓점
#     s = pts.sum(axis = 1)
#     points[0] = pts[np.argmin(s)] #좌상
#     points[3] = pts[np.argmax(s)] #우하

#     #원점 (0,0)은 맨 왼쪽 상단에 있으므로, y-x의 값이 가장 작으면 우상의 꼭짓점 / y-x의 값이 가장 크면 좌하의 꼭짓점
#     diff = np.diff(pts, axis = 1)
#     points[2] = pts[np.argmin(diff)] #우상
#     points[1] = pts[np.argmax(diff)] #좌하

#     src.append(points[0])
#     src.append(points[1])
#     src.append(points[2])
#     src.append(points[3])

#     return points



if image is not None:
    find_and_mask_numbers(image)
else:
    print("오류: 이미지를 불러올 수 없습니다.")

# def testImageOpenByPytesseract(image_path):
#     print(pytesseract.image_to_string(Image.open(image_path)))

# testImageOpenByPytesseract(image_path)