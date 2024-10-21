
import os
import cv2
import face_recognition
import numpy as np

# Directory for storing registered faces
REGISTERED_FACES_DIR = "registered_faces"
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

def register_face(name):
    """Capture and register a face with a given name."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Draw rectangles and grid pattern on detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw grid pattern
            for i in range(left, right, 10):
                cv2.line(frame, (i, top), (i, bottom), (255, 0, 0), 1)
            for j in range(top, bottom, 10):
                cv2.line(frame, (left, j), (right, j), (255, 0, 0), 1)

        cv2.imshow('Register Face - Press "s" to save, "q" to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Save the face encoding
            if face_encodings:
                np.save(os.path.join(REGISTERED_FACES_DIR, f"{name}.npy"), face_encodings[0])
                print(f"Registered {name}'s face.")
                break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Corrected function name

if __name__ == "__main__":
    name = input("Enter the name for registration: ")
    register_face(name)
