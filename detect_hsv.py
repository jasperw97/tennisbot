import cv2
import numpy as np

image = cv2.imread("image.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#HSV ranges
lower = np.array([42, 15, 160])
upper = np.array([58, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# transformed_bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
cv2.imshow("transformed", mask)
cv2.waitKey(0)