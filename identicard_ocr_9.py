import numpy as np
import cv2
import pytesseract
import os
from PIL import Image

# Step 1: Edge Detection
image = cv2.imread("./identicard.jpg")
orig = image.copy()

r = 800.0 / image.shape[0]
dim = (int(image.shape[1]*r), 800)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 75, 200)

print("STEP 1: Edge detection")

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Find Countours of Paper

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

print("STEP 2: Find contours of paper")

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

rect = order_points(screenCnt.reshape(4,2) / r)
(topLeft, topRight, bottomRight, bottomLeft) = rect

w1 = abs(bottomRight[0] - bottomLeft[0])
w2 = abs(topRight[0] - topLeft[0])
h1 = abs(topRight[1] - bottomRight[1])
h2 = abs(topLeft[1] - bottomLeft[1])

maxWidth = max([w1, w2])
maxHeight = max([h1, h2])
print('maxWidth', maxWidth)
print('maxHeight', maxHeight)

dst = np.float32([[0,0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])

M = cv2.getPerspectiveTransform(rect, dst)

warped = cv2.warpPerspective(orig, M, (100, 80))
# warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
print("STEP 3: Apply perspective transform")
# cv2.imshow("Warped", warped)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# STEP 4. Apply Adaptive Threshold

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

print("STEP 4. Apply Adaptive Threshold")

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, warped)

# text = pytesseract.image_to_string(Image.open(filename), lang = 'eng+kor')
text = pytesseract.image_to_string(warped, lang = 'eng+kor')
os.remove(filename)
print(text)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Scanned", cv2.WINDOW_NORMAL)
cv2.imshow("Original", orig)
cv2.imshow("Scanned", warped)
cv2.imwrite("ScannedImage.png", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()