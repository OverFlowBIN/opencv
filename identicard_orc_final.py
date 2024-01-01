import numpy as np
import cv2
from PIL import Image
import pytesseract

def order_points(pts):
    # 좌표를 정렬하기 위한 함수
    # 좌상단, 우상단, 우하단, 좌하단 순서로 정렬됨
    rect = np.zeros((4, 2), dtype="float32")

    # 좌상단 좌표는 합이 가장 작고, 우하단 좌표는 합이 가장 큰 좌표로 설정
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 이제 각 좌표 간의 차이를 계산, 우상단 좌표는 차이가 가장 작고, 좌하단 좌표는 차이가 가장 큰 좌표로 설정
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 정렬된 좌표 반환
    return rect

def auto_scan_image():
    # 이미지를 로드하고 기존 높이 대비 새로운 높이의 비율을 계산하여 크론하고 크기 조정
    # document.jpg ~ document7.jpg
    image = cv2.imread('assets/idcard_18.jpeg')
    orig = image.copy()
    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 이미지를 그레이스케일로 변환하고 블러 처리하고 에지 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 100, 200)

    # 원본 이미지와 에지 감지된 이미지 표시
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # 에지 이미지에서 윤곽선을 찾아 가장 큰 것들만 유지하고 스크린 윤곽선 초기화
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # 윤곽선들에 대해 반복
    for c in cnts:
        # 윤곽을 근사화
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 근사화한 윤곽이 4개의 점을 가지고 있다면, 화면을 찾은 것으로 간주
        if len(approx) == 4:
            screenCnt = approx
            break

    # 종이의 윤곽(외곽선)을 표시
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # 네 점 변환을 적용하여 원본 이미지의 위쪽에서 본 시점을 얻음
    rect = order_points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    
    dst = np.float32([[0,0], [maxWidth-1,0], 
                      [maxWidth-1,maxHeight-1], [0,maxHeight-1]])
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (int(maxWidth), int(maxHeight)))

    # 원본 이미지와 스캔된 이미지 표시
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Warped", warped)
    
    
    # Check the dimensions of the scanned image
    height, width = warped.shape[:2]
    
    isRotate = height > width
    if isRotate:
        warped = cv2.resize(warped, (540, 860), interpolation=cv2.INTER_AREA)    
        rotated_warped = cv2.transpose(warped)
        rotated_warped = cv2.flip(rotated_warped, flipCode=0)  

        # Display the rotated image
        print("STEP 4: Rotate the image")
        
        COLOR = (255, 255, 255)
        cv2.rectangle(rotated_warped, (290, 200), (450, 260), COLOR, cv2.FILLED) # 속이 꽉찬 사각형
        
        cv2.imshow("Rotated Scanned", rotated_warped)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    resized_warped = cv2.resize(warped, (860, 540), interpolation=cv2.INTER_AREA)    
    
    COLOR = (255, 255, 255)
    cv2.rectangle(resized_warped, (290, 200), (450, 260), COLOR, cv2.FILLED) # 속이 꽉찬 사각형
    
    cv2.imshow("Rotated Scanned", resized_warped)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # # 스캔된 이미지를 그레이스케일로 변환하고 쓰레시홀드를 적용하여 '흑백' 효과 부여
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    # # 원본 이미지와 스캔된 이미지 표시
    # print("STEP 4: Apply Adaptive Threshold")
    
    # cv2.imshow("Original", orig)
    # cv2.imshow("Scanned", warped)
    # # cv2.imwrite('scannedImage.png', warped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    # # Check the dimensions of the scanned image
    # height, width = warped.shape[:2]

    # # If the height is greater than the width, rotate the image by 90 degrees
    # isRotate = height > width
    # if isRotate:
    #     rotated_warped = cv2.transpose(warped)
    #     rotated_warped = cv2.flip(rotated_warped, flipCode=0)  

    #     # Display the rotated image
    #     print("STEP 5: Rotate the image")
    #     cv2.imshow("Rotated Scanned", rotated_warped)
        
    #     # cv2.imshow('rotated_warped', rotated_warped)
    #     # text = pytesseract.image_to_string(rotated_warped, lang='kor')
        
    #     # print("STEP 6: text")
    #     # print(text)  
        
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    
    
  
      
    
    
        
        
if __name__ == '__main__':
    auto_scan_image()
    
    