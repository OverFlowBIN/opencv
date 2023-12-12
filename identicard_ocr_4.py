import cv2
import numpy as np

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
    cv2.imshow('gray', gray)

    # 이미지를 수평으로 회전합니다.
    rotated_image = rotate_image(gray, 0)

    # 이미지 크기를 고정 크기로 조정합니다 (필요에 따라 조절).
    resized_image = resize_image(rotated_image, 600, 400)

    # 노이즈를 감소시키기 위해 가우시안 블러를 적용합니다.
    blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # EAST 텍스트 탐지기를 사용하여 텍스트 영역을 찾습니다.
    east_path = "path/to/east_text_detector.pb"  # 실제 EAST 텍스트 탐지기 모델 경로로 교체
    net = cv2.dnn.readNet(east_path)

    # 텍스트 감지를 위한 관심 영역(ROI)을 정의합니다.
    (H, W) = resized_image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # 예측을 디코딩하고 텍스트 영역을 추출합니다.
    rectangles, confidences = cv2.text.detect_boxes(scores, geometry)
    indices = cv2.dnn.NMSBoxesRotated(rectangles, confidences, 0.5, 0.4)

    # 가장 큰 직사각형 윤곽(정사각형 또는 직사각형)을 찾습니다.
    max_contour = None
    max_contour_area = 0

    for i in indices:
        # 회전된 직사각형을 추출합니다.
        box = cv2.boxPoints(rectangles[i[0]])
        box = np.int0(box)

        # 윤곽의 면적을 계산합니다.
        contour_area = cv2.contourArea(box)

        # 현재 윤곽의 면적이 최대 윤곽의 면적보다 크면 최대 윤곽을 업데이트합니다.
        if contour_area > max_contour_area:
            max_contour = box
            max_contour_area = contour_area

    # 가장 큰 직사각형 윤곽을 마스킹합니다.
    masked_image = resized_image.copy()
    if max_contour is not None:
        cv2.drawContours(masked_image, [max_contour], 0, 255, thickness=cv2.FILLED)

    # 원본 및 마스킹된 이미지를 표시합니다.
    cv2.imshow('원본 이미지', resized_image)
    cv2.imshow('마스킹된 이미지', masked_image)

    # 마스킹된 이미지를 저장합니다.
    cv2.imwrite('masked_image.jpg', masked_image)

    # 키 입력 대기 및 창 닫기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실제 이미지의 경로로 'your_image_path.jpg'를 대체하세요.
image_path = './identicard_2.jpg'
image = cv2.imread(image_path)

if image is not None:
    find_and_mask_numbers(image)
else:
    print("오류: 이미지를 불러올 수 없습니다.")