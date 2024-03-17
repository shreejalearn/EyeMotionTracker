import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import Counter

mp_face_mesh = mp.solutions.face_mesh
cap = cv.VideoCapture(0)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# List to store iris positions
left_iris_positions = []
right_iris_positions = []

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            left_iris = mesh_points[LEFT_IRIS]
            right_iris = mesh_points[RIGHT_IRIS]

            # Calculate iris positions
            left_iris_positions.append(tuple(left_iris.mean(axis=0).astype(int)))
            right_iris_positions.append(tuple(right_iris.mean(axis=0).astype(int)))

            # Draw iris circles
            cv.circle(frame, tuple(left_iris_positions[-1]), 2, (0, 255, 0), -1)
            cv.circle(frame, tuple(right_iris_positions[-1]), 2, (0, 255, 0), -1)

            # Calculate most frequent iris positions
            if len(left_iris_positions) > 10:  # adjust the number of frames to consider
                most_common_left = Counter(left_iris_positions).most_common(1)
                most_common_right = Counter(right_iris_positions).most_common(1)
                cv.circle(frame, most_common_left[0][0], 5, (255, 0, 0), -1)
                cv.circle(frame, most_common_right[0][0], 5, (255, 0, 0), -1)

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
