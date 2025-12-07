import cv2
import os

video_path = "tennis.MPG"
output_dir = "justin_serve_game"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break  # End of video
    output_filename = os.path.join(output_dir, f"frame_{str(frame_count).zfill(5)}.png")
    cv2.imwrite(output_filename, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

