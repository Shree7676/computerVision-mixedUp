import json

import cv2
import face_recognition
import numpy as np
from tensorflow.keras import models

import trained_models
from mtcnn.mtcnn import MTCNN

trained_model = models.load_model("./trained_models/trained_vggface.h5", compile=False)
trained_model.summary()
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
    7: " ",
}
detector = MTCNN()
# start the webcam feed
black = np.zeros((96, 96))

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
self_img = face_recognition.load_image_file("self4.png")
self_encoding = face_recognition.face_encodings(self_img)[0]


with open("details.json", "r") as file:
    data = json.load(file)

with open("encoding.json", "r") as file:
    encoded_data = json.load(file)

# Check and convert string representations to lists of floats if necessary
known_face_encodings = []
for encoding in encoded_data:
    if isinstance(encoding, list):
        known_face_encodings.append(np.array(encoding, dtype=np.float32))
    else:
        try:
            encoding_list = json.loads(encoding)
            known_face_encodings.append(np.array(encoding_list, dtype=np.float32))
        except json.JSONDecodeError:
            print(f"Error decoding encoding: {encoding}")

known_face_encodings.append(self_encoding)
data.append({"Name:": "Shree"})
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

maxindex = 7
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        results = detector.detect_faces(frame)
        if len(results) == 1:  # len 1 = face else no face
            x1, y1, width, height = results[0]["box"]
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = frame[y1:y2, x1:x2]

            cropped_img = cv2.resize(face, (96, 96))
            cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
            cropped_img_float = cropped_img_expanded.astype(float)
            prediction = trained_model.predict(cropped_img_float)
            print(prediction)
            maxindex = int(np.argmax(prediction))

            rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    print(data[best_match_index])
                    name = data[best_match_index]["Name:"]

                face_names.append(name)
    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED,
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame,
            name + ":" + emotion_dict[maxindex],
            (left + 6, bottom - 6),
            font,
            1.0,
            (255, 255, 255),
            1,
        )

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
