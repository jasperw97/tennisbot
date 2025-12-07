import cv2 
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
#   target_joints = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  for pose_landmarks in pose_landmarks_list:
        # Extract normalized coordinates
        xs = [lm.x for lm in pose_landmarks]
        ys = [lm.y for lm in pose_landmarks]

        # Convert to pixel coordinates
        H, W, _ = rgb_image.shape
        x_min, x_max = int(min(xs)*W) - 30, int(max(xs)*W) + 30
        y_min, y_max = int(min(ys)*H) - 30, int(max(ys)*H) + 30

        # Draw rectangle
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
  return annotated_image

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
    while True:
        success, frame = cap.read()
        # scale_factor = 5
        # frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        time = frame_count / fps
        timestamp = mp.Timestamp.from_seconds(time)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp.value)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        annotated_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("annotated", annotated_frame)
        frame_count += 1
        if cv2.waitKey(10) == ord('q'):
            break