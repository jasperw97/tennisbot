import numpy as np
import cv2 

image = cv2.imread("image.png")

# Method 1 my proposed ball filter for convolution, didn't quite work

# Method 2 Yellow Plane Extraction from the paper, also didn't quite work because ball might not always be yellow in camera
# CV2 uses b, g, r
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur_image = cv2.GaussianBlur(image, (7, 7), 0)
# calc_image = np.copy(blur_image)
# calc_image = calc_image.astype(np.float32)
# yellow_extraction = calc_image[:, :, 1] - calc_image[:, :, 2]/1.45 - calc_image[:, :, 0]/1.45
# yellow_extraction = np.where(yellow_extraction > 10, 255, 0)
# yellow_extraction = yellow_extraction.astype(np.uint8)

# row, col = np.where(yellow_extraction == 255)

# for i in range(len(row)):
#     cv2.circle(image, (row[i], col[i]), 2, color=(0, 255, 0), thickness=2)


#Method 3 Hough Circles in opencv
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0)
edges = cv2.Canny(blur, 50, 150)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,  # Inverse ratio of accumulator resolution
     minDist=100,  # Minimum distance between centers of detected circles
     param1=150,  # Upper threshold for the internal Canny edge detector
     param2=15,  # Threshold for center detection
     minRadius=1,  # Minimum circle radius to detect
     maxRadius=50)

circles = circles[0, :]
print(circles)
for circle in circles:
    cv2.circle(image, (int(round(circle[1])), int(round(circle[0]))), int(round(circle[2])), color=(0, 255, 0), thickness=10)
cv2.imshow("edges", edges)
cv2.waitKey(0)
# print(image.shape)
# cv2.imshow("IMG", image)
cv2.waitKey(0)