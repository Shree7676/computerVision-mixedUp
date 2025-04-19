import cv2
import face_recognition
import numpy as np
from tensorflow.keras import models

from mtcnn.mtcnn import MTCNN

# Load the emotion recognition model
trained_model = models.load_model("./trained_models/trained_vggface.h5", compile=False)

# Prevents unnecessary logging and optimizes OpenCV usage
cv2.ocl.setUseOpenCL(False)

# Emotion label dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# MTCNN for face detection
detector = MTCNN()

# Start the webcam feed
video_capture = cv2.VideoCapture(0)

# Load sample images and create known encodings
self_img = face_recognition.load_image_file("self4.png")
self_encoding = face_recognition.face_encodings(self_img)[0]

biden_image = face_recognition.load_image_file("biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [self_encoding, biden_face_encoding]
known_face_names = ["Shree", "Joe Biden"]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
displayed_name = ""
displayed_emotion = ""
displayed_box = (0, 0, 0, 0)
frame_count = 0  # Frame counter to control processing frequency
stable_frame_count = 5  # Number of frames to hold the display stable

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Only process every few frames (e.g., every third frame) to reduce flickering
    if frame_count % 3 == 0:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces in the frame using face_recognition
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        # Variables to store current frame's detected face and emotion
        current_names = []
        current_emotion = ""
        current_box = (0, 0, 0, 0)

        if face_locations:
            # Process only the first detected face
            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                # Check for the best match
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    # Scale up face coordinates
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                    current_box = (left, top, right, bottom)

                    # Extract face region for emotion detection
                    face = frame[top:bottom, left:right]
                    if face.size > 0:
                        cropped_img = cv2.resize(face, (96, 96))
                        cropped_img = np.expand_dims(cropped_img, axis=0).astype(
                            "float32"
                        )
                        prediction = trained_model.predict(cropped_img)
                        maxindex = int(np.argmax(prediction))
                        current_emotion = emotion_dict[maxindex]

                current_names.append(name)

            # Update displayed values if they remain stable for a few frames
            if current_names and (
                displayed_name != current_names[0]
                or displayed_emotion != current_emotion
            ):
                displayed_name = current_names[0]
                displayed_emotion = current_emotion
                displayed_box = current_box
                frame_count = 0  # Reset the stable frame counter

    frame_count += 1

    # Display the last stable box, name, and emotion
    left, top, right, bottom = displayed_box
    if displayed_name:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw the name and emotion label
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        label = f"{displayed_name} - {displayed_emotion}"
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Video", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
