
import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'alpaca.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(
    *"mp4v"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

model_path = os.path.join('.', 'kaggle', 'working', 'runs', 'detect',
                          'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.5
class_name_dict = {0: 'alpaca'}

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    # cv2.imshow('dsf', frame)

    cv2.waitKey(1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
