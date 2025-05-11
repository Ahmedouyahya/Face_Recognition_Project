#!/usr/bin/env python3
import os
import pickle
import logging
import cv2
import face_recognition

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "recognition.log")
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s - %(message)s",
        level=logging.INFO
    )
    print(f"[OK] Logging to {log_path}")

def load_encodings(encodings_path="models/encodings.pkl"):
    if not os.path.exists(encodings_path):
        raise FileNotFoundError(f"Run train.py first to create {encodings_path}")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_from_webcam(known_encs, known_names, tolerance=0.6):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("[INFO] Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR→RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        # Loop over each face found
        for (top, right, bottom, left), enc in zip(locations, encodings):
            matches = face_recognition.compare_faces(known_encs, enc, tolerance)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                logging.info(f"{name} recognized")

            # Draw bounding box and name label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        # Display result
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")

if __name__ == "__main__":
    setup_logging()
    known_encodings, known_names = load_encodings()
    recognize_from_webcam(known_encodings, known_names)
