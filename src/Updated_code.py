import cv2
import face_recognition
import numpy as np
from tensorflow.keras import models

# Load the emotion recognition model
trained_model = models.load_model("./trained_models/trained_vggface.h5", compile=False)

# Prevents unnecessary logging and optimizes OpenCV usage
cv2.ocl.setUseOpenCL(False)

# Dictionary mapping emotion indices to labels
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Initialize the webcam feed
video_capture = cv2.VideoCapture(0)

# Load and encode sample images for face recognition
self_img = face_recognition.load_image_file("self4.png")
self_encoding = face_recognition.face_encodings(self_img)[0]

biden_image = face_recognition.load_image_file("biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Known faces and their labels
known_face_encodings = [self_encoding, biden_face_encoding]
known_face_names = ["Shree", "Joe Biden"]

# Variables to handle frame processing
process_this_frame = True
maxindex = 6  # Default to "Neutral" if no emotion detected

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process only every other frame
    if process_this_frame:
        # Resize the frame for faster processing and convert color for face_recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Face recognition: detect faces and encode them
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # See if the face is a match for known faces
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # Use the closest match if one is found
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Scale up face locations to original frame size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Extract face area for emotion detection
                face = frame[top:bottom, left:right]
                if face.size > 0:
                    # Resize and prepare for emotion model
                    cropped_img = cv2.resize(face, (96, 96))
                    cropped_img = np.expand_dims(cropped_img, axis=0).astype("float32")
                    prediction = trained_model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                else:
                    maxindex = 6  # Default to "Neutral"

            face_names.append((name, emotion_dict[maxindex]))

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), (name, emotion) in zip(face_locations, face_names):
        # Scale up face locations to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name and emotion
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        label = f"{name} - {emotion}"
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
