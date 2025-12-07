"""
Functions that perform non-maximal suppression to remove duplicate detections

Idea: sort detections by box size: start with the largest box -> loop through the array, if any
other detections overlap with it (even just a little bit), remove the smaller detection, this is 
desired in our application because we want to track tennis ball and minimize all other noise when
possible
"""

def has_overlap(detect1, detect2):
    """
    Docstring for has_overlap
    
    :param detect1: [x, y, w, h] of detection 1
    :param detect2: [x, y, w, h] of detection 2
    """
    result = True
    if (detect1[0] + detect1[2]) < detect2[0] or (detect2[0] + detect2[2]) < detect1[0]:
        result = False
    
    if (detect1[1] + detect1[3]) < detect2[1] or (detect2[1] + detect2[3]) < detect1[1]:
        result = False
    
    return result

def non_maximal_suppression(detections):
    """
    Docstring for non_maximal_suppression
    
    :param detections: list of detections with each element being [area, [x, y, w, h]]
    """
    detections_sorted = sorted(detections, key=lambda item: item[0], reverse=True)
    
    i = 0
    while i < len(detections_sorted):
        j = i + 1
        while j < len(detections_sorted):
            if has_overlap(detections_sorted[i][1], detections_sorted[j][1]):
                del detections_sorted[j]  # safe because j moves forward
            else:
                j += 1
        i += 1
    
    return detections_sorted
    