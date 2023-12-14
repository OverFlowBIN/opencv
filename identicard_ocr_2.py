import cv2
import numpy as np
import pytesseract
from PIL import Image
# from random import random

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
image_path = './identicard_3.jpg'
# image_path = './test.png'
image = cv2.imread(image_path)


def find_and_mask_numbers(image):

    resized_image = resize_image(image, 1200, 900)


    # 이미지를 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)
    # cv2.imshow('gray', gray)


    # 노이즈를 감소시키고 윤곽을 감지하기 위해 가우시안 블러를 적용합니다.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('blurred', blurred)

    # Canny를 사용하여 가장자리를 검출합니다.
    edges = cv2.Canny(blurred, 75, 200, True)
    # cv2.imshow('edges', edges)

    # 가장자리 감지된 이미지에서 윤곽을 찾습니다.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    drawContours = cv2.drawContours(resized_image, contours, -1, (0, 0, 225), 3)
    # cv2.imshow('drawContours', drawContours)
    print('contours_1', len(contours))

    # cnts = imutils.grab_contours(cnts)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    print('contours_2', len(contours))

    # for i in contours:
    #     for j in i:
    #         cv2.circle(resized_image, tuple(j[0]), 1, (255,255,0), -1)

    # cv2.imshow('cv2.circle', resized_image)

    # x, y, w, h = cv2.boundingRect(contours[2])


    # rectangle = cv2.rectangle(resized_image, (x,y), (x+w, y+h), (0, 255, 0), 3)

    # cv2.imshow('rectangle', rectangle)

    # rect  = cv2.minAreaRect(contours)
    # cv2.imshow('rect', rect)

    # ===========================

    # arr = [];

    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # for contour in contours:
    #     epsilon = 0.1 * cv2.arcLength(contour, True)
    #     # print('epsilon', epsilon)
    #     approx = cv2.approxPolyDP(contour, epsilon, True)
    #     # print('approx', approx)



    #     if len(approx) == 4:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         arr.append([x, y, w, h])

    #         rectangle = cv2.rectangle(resized_image, (x,y), (x+w, y+h), (255, 0, 0), 3)
    #         cv2.imshow('rectangle', rectangle)

    #         break


    # print('arr', arr)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ===========================


    # drawContours = cv2.drawContours(edges, contours, 0, (255, 0, 0), 3)
    # # cv2.imshow('drawContours', drawContours)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # 면적을 기준으로 윤곽을 내림차순으로 정렬합니다.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # print('contours', contours)

    # 윤곽을 반복하여 가장 큰 직사각형 윤곽을 찾습니다 (ID 카드로 가정).
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # print('len(approx)', len(approx))
        print('approx', approx)

        # x, y, w, h = cv2.boundingRect(contour)

        if len(approx) == 4:
            # 직사각형 윤곽을 찾음 (ID 카드로 가정).
            x, y, w, h = cv2.boundingRect(contour)
            print('x', x)
            print('y', y)
            print('w', w)
            print('h', h)


            cv2.circle(resized_image, (x, y), 2, (255,255,0), 2)
            cv2.circle(resized_image, (x + w, y), 2, (255,255,0), 2)
            cv2.circle(resized_image, (x, y + h), 2, (255,255,0), 2)
            cv2.circle(resized_image, (x + w, y + h), 2, (255,255,0), 2)

            cv2.imshow('cv2.circle', resized_image)

            # extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            # extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            # extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            # extBot = tuple(contour[contour[:, :, 1].argmax()][0])
            # print('extLeft', extLeft)
            # print('extRight', extRight)
            # print('extTop', extTop)
            # print('extBot', extBot)

            # cv2.circle(resized_image, extLeft, 8, (0, 0, 255), -1)
            # cv2.circle(resized_image, extRight, 8, (0, 255, 0), -1)
            # cv2.circle(resized_image, extTop, 8, (255, 0, 0), -1)
            # cv2.circle(resized_image, extBot, 8, (255, 255, 0), -1)

            # cv2.imshow('cv2.circle', resized_image)



            # break


            # ID 카드 영역을 수평으로 회전시킵니다.
            # rotated_image = rotate_image(image[y:y + h, x:x + w], 0)



            # 이미지 크기를 고정 크기로 조정합니다 (필요에 따라 조절).
            # resized_image = resize_image(rotated_image, 600, 400)

            # 크기를 조정한 이미지를 그레이스케일로 변환합니다.
            # gray_resized = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)


            # OCR 또는 다른 방법을 사용하여 주민등록번호를 찾아 마스킹합니다.
            # 여기서는 OCR로 pytesseract를 사용한다고 가정합니다.
            # try:
            #     # import pytesseract

            #     # pytesseract를 사용하여 텍스트 추출
            #     text = pytesseract.image_to_string(gray_resized, config='--psm 6')


            #     # 주민등록번호를 마스킹합니다.
            #     masked_image = rotated_image.copy()

            #     for word in text.split():
            #         if '-' in word and len(word) == 14:
            #             # 숫자를 마스킹하기 위해 채워진 사각형을 그립니다.
            #             cv2.putText(masked_image, '*' * 14, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #     # 원본 및 마스킹된 이미지를 표시합니다.
            #     cv2.imshow('원본 이미지', rotated_image)
            #     cv2.imshow('마스킹된 이미지', masked_image)

            #     # 마스킹된 이미지를 저장합니다.
            #     cv2.imwrite('masked_image.jpg', masked_image)

            #     # 키 입력 대기 및 창 닫기
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            # except ImportError:
            #     print("pytesseract 모듈이 설치되어 있지 않습니다. 'pip install pytesseract'를 사용하여 설치하십시오.")

            # break

            cv2.waitKey(0)
            cv2.destroyAllWindows()




if image is not None:
    find_and_mask_numbers(image)
else:
    print("오류: 이미지를 불러올 수 없습니다.")

# def testImageOpenByPytesseract(image_path):
#     print(pytesseract.image_to_string(Image.open(image_path)))

# testImageOpenByPytesseract(image_path)