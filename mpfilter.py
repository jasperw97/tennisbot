import cv2 
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functions import get_mask, mp_filter


def extract_boxes(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  if not pose_landmarks_list:
    return None, None, None, None
  
  for pose_landmarks in pose_landmarks_list:
        # Extract normalized coordinates
        xs = [lm.x for lm in pose_landmarks]
        ys = [lm.y for lm in pose_landmarks]

        # Convert to pixel coordinates
        H, W, _ = rgb_image.shape
        x_min, x_max = int(min(xs)*W) - 30, int(max(xs)*W) + 30
        y_min, y_max = int(min(ys)*H) - 30, int(max(ys)*H) + 30
  return x_min, x_max, y_min, y_max
  

model_path = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture("tennis.MPG")

if not cap.isOpened():
    print("Error, can't open video file")
else:
    print("Video opened successfully")
    fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    prev_frame = None
    while True:
        success, frame = cap.read()
        bgr_frame = frame.copy()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        time = frame_count / fps
        timestamp = mp.Timestamp.from_seconds(time)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp.value)
        xmin, xmax, ymin, ymax = extract_boxes(mp_image.numpy_view(), pose_landmarker_result)
        
        
        if prev_frame is not None and xmin is not None:
          mask = get_mask(prev_frame, bgr_frame)
          copy = mp_filter(mask, bgr_frame, xmin, xmax, ymin, ymax)
          cv2.imshow("filtered", copy)
        
        prev_frame = bgr_frame
        # cv2.imshow("annotated", annotated_frame)
        frame_count += 1
        if cv2.waitKey(10) == ord('q'):
            break