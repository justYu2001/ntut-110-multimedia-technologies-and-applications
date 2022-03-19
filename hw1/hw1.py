import numpy as np
import cv2

image = cv2.imread('road.jpg')
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', imageGray)

blured = cv2.GaussianBlur(imageGray, (3, 3), 0)
cv2.imwrite('blur.jpg', blured)

_, otsuThreshold = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('threshold.jpg', otsuThreshold)

closeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closeOp = cv2.morphologyEx(otsuThreshold, cv2.MORPH_CLOSE, closeKernel)
cv2.imwrite('closeOp.jpg', closeOp)

canny = cv2.Canny(closeOp, 100, 200)
cv2.imwrite('canny.jpg', canny)

hough = cv2.HoughLinesP(canny, 2.0, np.pi/180, 150, minLineLength=150, maxLineGap=100)

result = np.copy(image)
for line in hough:
  for x1, y1, x2, y2 in line:
    if y1 < 400 and y2 < 400: continue
    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imwrite('output.jpg', result)