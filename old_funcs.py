import cv2
from functions import get_mask
import numpy as np

def detect_circle(edges, original_frame):
    """
    Docstring for detect_circle
    
    :param edges: 2d np array of canny edges
    """
    copy = original_frame.copy()
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=15, minRadius=1, maxRadius=15)
    if circles is not None:
        circles = circles[0, :]
        for circle in circles:
            cv2.circle(copy, (int(round(circle[1])), int(round(circle[0]))), int(round(circle[2])), color=(0, 255, 0), thickness=3)
    return copy

def get_edges(frame1, frame2, kernel=np.array((3, 3), dtype=np.uint8)):
    mask = get_mask(frame1, frame2, kernel=np.array((3, 3), dtype=np.uint8))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(mask, 50, 150)
    return edges

def contour_detect_and_draw(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 2)
    return copy

def contour_fit_ellipse(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            if MA > 1 and ma > 1 and (abs(1 - MA/ma) < 0.2):  # skip degenerate ellipses
                cv2.ellipse(copy, ellipse, (0,255,0), 2)
            
    
    return copy

def mp_filter(mask, original_frame, xmin, xmax, ymin, ymax):
    copy = original_frame.copy()
    centers = []
    radiuses = []
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        centers.append([x, y])
        radiuses.append(r)
    centers = np.array(centers)
    radiuses = np.array(radiuses)
    # Filter out centers that are in the body area
    mask = (centers[:, 0] > xmax) & (centers[:, 0] < xmin) & (centers[:, 1] > ymax) & (centers[:, 1] < ymin)
    centers = centers[mask]
    radiuses = radiuses[mask]
    i = 0
    for center in centers:
        cv2.circle(copy, center, radiuses[i], (0, 255, 0), 3)
        i += 1
    return copy

def contour_isolation_filter(mask, original_frame, dist_threshold):
    copy = original_frame.copy()
    centers = []
    radiuses = []
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        centers.append([x, y])
        radiuses.append(r)
    centers = np.array(centers)

    index = 0
    for center in centers:
        distances = np.linalg.norm(centers - center, axis=1)
        distances = distances[distances != 0]
        distances = distances[distances <= dist_threshold]
        if len(distances) < 2 and radiuses[index] < 5:
            cv2.circle(copy, (int(center[0]), int(center[1])), int(radiuses[index]), (0, 0, 255), 3)
        index += 1

    return copy

def get_diff(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    #Grayscale Conversion
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.subtract(gray2, gray1)
    diff = cv2.medianBlur(diff, 3)

    return diff