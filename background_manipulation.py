import cv2
import numpy as np

court = cv2.imread("image.png")
bg = cv2.imread("background.png")
h, w = court.shape[:2]
bg = cv2.resize(bg, (w, h))

cv2.imshow("court", court)
cv2.imshow("bg", bg)
cv2.waitKey(0)