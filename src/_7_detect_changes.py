import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh and Drawing Utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Constants for thresholding and landmarks
LEFT_EYE_LANDMARKS = [145, 159]
RIGHT_EYE_LANDMARKS = [374, 386]
LEFT_EYEBROW_LANDMARKS = [70, 63]
RIGHT_EYEBROW_LANDMARKS = [300, 293]
MOUTH_LANDMARKS = [13, 14]
HEAD_LEFT_LANDMARKS = [234, 93]
HEAD_RIGHT_LANDMARKS = [454, 323]

# Thresholds for different gestures
EYE_ASPECT_RATIO_THRESHOLD = 0.2
EYEBROW_DISTANCE_THRESHOLD = 0.1
MOUTH_OPEN_THRESHOLD = 0.15
HEAD_TILT_THRESHOLD = 0.15

# Main loop
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # If landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to numpy array for easier processing
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )

                # Compute reference distances for normalization
                face_width = np.linalg.norm(
                    landmarks[HEAD_LEFT_LANDMARKS[0]]
                    - landmarks[HEAD_RIGHT_LANDMARKS[0]]
                )

                # Left and Right Eye Blink Detection
                def is_eye_closed(eye_landmarks):
                    return (
                        np.linalg.norm(
                            landmarks[eye_landmarks[0]] - landmarks[eye_landmarks[1]]
                        )
                        / face_width
                        < EYE_ASPECT_RATIO_THRESHOLD
                    )

                left_eye_closed = is_eye_closed(LEFT_EYE_LANDMARKS)
                right_eye_closed = is_eye_closed(RIGHT_EYE_LANDMARKS)

                # Mouth Open Detection
                mouth_open = (
                    np.linalg.norm(
                        landmarks[MOUTH_LANDMARKS[0]] - landmarks[MOUTH_LANDMARKS[1]]
                    )
                    / face_width
                    > MOUTH_OPEN_THRESHOLD
                )

                # Eyebrow Raise Detection
                def is_eyebrow_raised(eye_landmark, eyebrow_landmark):
                    return (
                        np.linalg.norm(
                            landmarks[eye_landmark] - landmarks[eyebrow_landmark]
                        )
                        / face_width
                        > EYEBROW_DISTANCE_THRESHOLD
                    )

                left_eyebrow_up = is_eyebrow_raised(
                    LEFT_EYE_LANDMARKS[0], LEFT_EYEBROW_LANDMARKS[0]
                )
                right_eyebrow_up = is_eyebrow_raised(
                    RIGHT_EYE_LANDMARKS[0], RIGHT_EYEBROW_LANDMARKS[0]
                )
                both_eyebrows_up = left_eyebrow_up and right_eyebrow_up

                # Head Tilt Detection
                head_tilt_left = (
                    landmarks[HEAD_LEFT_LANDMARKS[0]][1]
                    - landmarks[HEAD_RIGHT_LANDMARKS[0]][1]
                ) / face_width > HEAD_TILT_THRESHOLD
                head_tilt_right = (
                    landmarks[HEAD_RIGHT_LANDMARKS[0]][1]
                    - landmarks[HEAD_LEFT_LANDMARKS[0]][1]
                ) / face_width > HEAD_TILT_THRESHOLD

                # Display detections on screen
                text = []
                if left_eye_closed:
                    text.append("Left Eye Blink")
                if right_eye_closed:
                    text.append("Right Eye Blink")
                if mouth_open:
                    text.append("Mouth Open")
                if left_eyebrow_up:
                    text.append("Left Eyebrow Up")
                if right_eyebrow_up:
                    text.append("Right Eyebrow Up")
                if both_eyebrows_up:
                    text.append("Both Eyebrows Up")
                if head_tilt_left:
                    text.append("Head Tilt Left")
                if head_tilt_right:
                    text.append("Head Tilt Right")

                cv2.putText(
                    frame,
                    ", ".join(text),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Draw the face mesh on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    ),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                )

        # Show the frame
        cv2.imshow("Facial Gesture Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
