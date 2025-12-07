import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("tennis2.MPG")

if not cap.isOpened():
    print("Error, can't open video file")
else:
    print("Video opened successfully")

while True:
    success, frame = cap.read()
    scale_factor = 2
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(
    gray,
    winStride=(8,8),    # step size for sliding window
    padding=(8,8),       # padding around window
    scale=1.05   # pyramid scale factor
    )

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("HOG Person Detection", frame)
    if cv2.waitKey(10) == ord('q'):
            break