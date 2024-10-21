




import os
import cv2
import face_recognition
import numpy as np

REGISTERED_FACES_DIR = "registered_faces"

def recognize_faces():
    known_face_encodings = []
    known_face_names = []

    # Load registered faces
    for filename in os.listdir(REGISTERED_FACES_DIR):
        if filename.endswith(".npy"):
            name = os.path.splitext(filename)[0]
            encoding = np.load(os.path.join(REGISTERED_FACES_DIR, filename))
            known_face_encodings.append(encoding)
            known_face_names.append(name)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=30)  # Increased num_jitters

        # Draw rectangles around recognized faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)  # Get distances
            name = "Unknown"

            # Check if the face is recognized
            if np.any(distances < 0.4):  # Adjust threshold
                first_match_index = np.argmin(distances)
                name = known_face_names[first_match_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face Recognition - Press "q" to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()

