#!/usr/bin/env python3
import os
import pickle
import cv2
import face_recognition
import logging

# Optional: log warnings to console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def encode_known_faces(known_dir="known_faces", model_dir="models"):
    encodings = []
    names = []

    # Walk through each person folder
    for person in os.listdir(known_dir):
        person_path = os.path.join(known_dir, person)
        if not os.path.isdir(person_path):
            continue

        # Process each image file
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            # Load with OpenCV
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                logging.warning(f"Cannot read image: {img_path}")
                continue

            # Convert BGR→RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Extract face encodings
            face_encs = face_recognition.face_encodings(rgb_img)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(person)
                logging.info(f"Encoded {person} from {img_name}")
            else:
                logging.warning(f"No face found in {img_name}")

    # Ensure output directory exists
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "encodings.pkl")

    # Serialize encodings + names
    with open(out_path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"[OK] Encodings saved to {out_path}")

if __name__ == "__main__":
    encode_known_faces()
