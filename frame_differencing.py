import cv2
import numpy as np
import matplotlib.pyplot as plt

#Reading images
img1 = cv2.imread("justin_serve_game/frame_00022.png")
img2 = cv2.imread("justin_serve_game/frame_00025.png")

#Grayscale Conversion
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Frame differencing
diff = cv2.subtract(gray2, gray1)
diff = cv2.medianBlur(diff, 3)
#Creating Mask fom the difference
mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 7)
mask = cv2.medianBlur(mask, 3)

# Morphological Closing operation
kernel = np.array((9, 9), dtype=np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
# print(mask)
# for row in mask:
#     print(max(row))
plt.imshow(mask, cmap="gray")
plt.show()