import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from nms import non_maximal_suppression, has_overlap

def get_mask(frame1, frame2, kernel=np.ones((3, 3), dtype=np.uint8)):
    #Grayscale Conversion
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray2, gray1)
    diff = cv2.medianBlur(diff, 3)

    #Creating Mask fom the difference
    # mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 7)
    retval, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # mask = cv2.medianBlur(mask, 3)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def contour_custom_draw(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        if r < 5:
            cv2.circle(copy, (int(x), int(y)), int(r), (0, 0, 255), 2)
            
    return copy

def get_valid_contours(mask, extended_roi):
    """
    Docstring for get_valid_contours
    
    :param mask: binary motion mask
    :param extended_roi: [x1, y1, x2, y2]
    => list of [x, y, w, h] detects (not rounded to int)
    """
    valid_detects = []
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if has_overlap([x, y, w, h], [extended_roi[0], extended_roi[1], (extended_roi[2] - extended_roi[0]), (extended_roi[3] - extended_roi[1])]):
            x, y, w, h = int(x), int(y), int(w), int(h), 
            valid_detects.append([x, y, w, h])

    return valid_detects            
        
def reformat(cv2detect):
    """
    Docstring for reformat
    
    :param cv2detect: [x, y, w, h]
    => [x1, y1, x2, y2]
    """
    return [cv2detect[0], cv2detect[1], cv2detect[0] +cv2detect[2], cv2detect[1] + cv2detect[3]]

def update_roi(roi, velocity):
    roi[0], roi[2] = roi[0] + velocity[0], roi[2] + velocity[0]
    roi[1], roi[3] = roi[1] + velocity[1], roi[3] + velocity[1]

def contour_draw_rect(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        # if area > 15:
        cv2.rectangle(copy, (int(x) - 1, int(y) - 1), (int(x+w) + 1, int(y+h) + 1), (0, 255, 0), 1)
    
    return copy

def nms_processed(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = int(x) - 20, int(y) - 20, int(w) + 40, int(h) + 40
        area = w*h
        detections.append([area, [x, y, w, h]])
    
    if len(detections) > 0:
        processed_results = non_maximal_suppression(detections)
        if len(processed_results) < 5:
            for result in processed_results:
                coords = result[1]
                cv2.rectangle(copy, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 1)
    
    return copy

def nms_with_size_select(mask, original_frame):
    copy = original_frame.copy()
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = int(x) - 5, int(y) - 5, int(w) + 5, int(h) + 5
        area = w*h
        detections.append([area, [x, y, w, h]])
    
    if len(detections) > 0:
        processed_results = non_maximal_suppression(detections)
        p_sorted  = sorted(processed_results, key=lambda item: item[0])
        i = 0
        # for result in p_sorted:
        #     if i > 3:
        #         break
        #     coords = result[1]
        #     cv2.rectangle(copy, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 1)
        #     i += 1
        coords = p_sorted[0][1]
        cv2.rectangle(copy, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 1)
    
    return copy

def get_ring_mean(gray_frame, x, y, w, h, pad=None):
    """
    Docstring for get_ring
    
    :param arr: grayscale image (2d numpy array)
    :param x: starting x for blob
    :param y: starting y for blob
    :param w: width of blob
    :param h: height of blob
    :param pad: padding for the ring
    => ring mean 
    """
    H, W = gray_frame.shape
    if pad == None:
        pad = 5 #Should Find a way to make this proportional to the size of the blob
    
    #Starting points for the bigger blob
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)
    x3 = min(W, x + w + pad)
    y3 = min(H, y + h + pad)
    
    expanded_roi = gray_frame[y2:y3, x2:x3]
    mask = np.ones_like(expanded_roi)
    mask[(y - y2):(y - y2 + h), (x - x2):(x - x2 + w)] = 0 #Filter out inner region
    
    outer_pixels = expanded_roi[mask.astype(bool)]
    return np.mean(outer_pixels)

def get_area_mean(gray_frame, x, y, w, h):
    """
    Docstring for get_area_mean
    
    the parameters for the area has to be pre processed to become integers
    
    :param gray_frame: gray frame
    :param x: starting x for blob
    :param y: starting y for blob
    :param w: width of blob
    :param h: height of blob
    """
    roi = gray_frame[y:y+h+1, x:x+w+1]
    return np.mean(roi)

def get_roi(frame, x, y, w, h):
    return frame[y:y+h+1, x:x+w+1]
    

def contour_contrast_filter(mask, original_frame):
    roi = None
    copy = original_frame.copy()
    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        #Update borders to enclose a larger area
        x, y, w, h = int(x) - 1, int(y) - 1, int(w) + 1, int(h) + 1
        
        #Filter out the large blobs and start contrast filtering
        if area < 60:
            ring_mean = get_ring_mean(gray, x, y, w, h)
            inner_mean = get_area_mean(gray, x, y, w, h)
            mean_diff = inner_mean - ring_mean
            roi = get_roi(copy, x, y, w, h)
            cnt_list.append([mean_diff, [x, y, w, h]])
            
    
    if len(cnt_list) != 0:
        cnt_list_sorted = sorted(cnt_list, key=lambda item: item[0], reverse=True)
        #Top 5 Version
        # for i in range(min(5, len(cnt_list_sorted))):
        #     winner = cnt_list_sorted[i][1]
        #     cv2.rectangle(copy, (winner[0], winner[1]), (winner[0] + winner[2], winner[1] + winner[3]), (0, 255, 0), 1)
        
        #Winner only version
        winner = cnt_list_sorted[0][1]
        cv2.rectangle(copy, (winner[0] - 2, winner[1] - 2), (winner[0] + winner[2] + 2, winner[1] + winner[3] + 2), (0, 255, 0), 2)
    
    return copy, roi


def get_mask_tune(frame1, frame2, kernel=np.ones((3, 3), dtype=np.uint8)):
    #Grayscale Conversion
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray2, gray1)
    diff = cv2.medianBlur(diff, 3)

    #Creating Mask fom the difference
    # mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 7)
    retval, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # mask = cv2.medianBlur(mask, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    

    return mask