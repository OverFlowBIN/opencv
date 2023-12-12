import numpy as np
import cv2 as cv
# img = cv.imread('./identicard.jpg',0)
img = cv.imread('./test.png',0)


ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print( M )

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

area = cv.contourArea(cnt)
perimeter = cv.arcLength(cnt,True)

epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)

hull = cv.convexHull(cnt)

k = cv.isContourConvex(cnt)

x,y,w,h = cv.boundingRect(cnt)
print('cv.boundingRect(cnt)', cv.boundingRect(cnt))
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.intp(box)
drawContours = cv.drawContours(img, [box], 0, (0,0,255), 2)
cv.imshow('drawContours', drawContours)


cv.waitKey(0)
cv.destroyAllWindows()