import cv2
import numpy as np
import pytesseract

def rotate_image(image, angle):
    # 지정된 각도로 이미지를 회전합니다.
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def resize_image(image, width, height):
    # 이미지 크기를 지정된 폭과 높이로 조정합니다.
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def find_and_mask_numbers(image):
    # 이미지를 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지를 수평으로 회전합니다.
    rotated_image = rotate_image(gray, 0)

    # 이미지 크기를 고정 크기로 조정합니다 (필요에 따라 조절).
    resized_image = resize_image(rotated_image, 600, 400)

    # 노이즈를 감소시키기 위해 가우시안 블러를 적용합니다.
    blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # pytesseract를 사용하여 텍스트를 추출합니다.
    text = pytesseract.image_to_string(blurred, config='--psm 6')

    # 감지된 주민등록번호를 마스킹합니다.
    masked_image = resized_image.copy()
    for word in text.split():
        if '-' in word and len(word) == 14:
            # 숫자를 마스킹하기 위해 채워진 사각형을 그립니다.
            cv2.putText(masked_image, '*' * 14, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 원본 및 마스킹된 이미지를 표시합니다.
    cv2.imshow('원본 이미지', resized_image)
    cv2.imshow('마스킹된 이미지', masked_image)

    # 마스킹된 이미지를 저장합니다.
    cv2.imwrite('masked_image.jpg', masked_image)

    # 키 입력 대기 및 창 닫기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실제 이미지의 경로로 'your_image_path.jpg'를 대체하세요.
image_path = './identicard.jpg'
image = cv2.imread(image_path)

if image is not None:
    find_and_mask_numbers(image)
else:
    print("오류: 이미지를 불러올 수 없습니다.")