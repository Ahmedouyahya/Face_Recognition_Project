# Face Recognition Project

A simple, real-time face recognition system in Python that:

* **Encodes** known faces from a directory of images.
* **Recognizes** faces in live webcam video.
* **Logs** recognized names with timestamps.

---

## 🚀 Features

* **Training**: Scan `known_faces/` subdirectories, extract 128-D face embeddings, and cache to `models/encodings.pkl`.
* **Recognition**: Detect and identify faces from your webcam in real time.
* **Logging**: Append timestamped recognition events to `logs/recognition.log`.
* **Robust loading**: Handles various JPEG formats by converting all images to standard 8-bit RGB arrays.

---

## 📦 Prerequisites

* **Python 3.12**
* **Webcam** for real-time recognition
* **Virtual environment** recommended

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/ageitgey/face_recognition_project.git
   cd face_recognition_project
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install face_recognition             # The world’s simplest face recognition API
   pip install opencv-python                # OpenCV for video I/O & drawing
   pip install pillow                       # Image loading & conversion
   pip install numpy                        # Numerical operations
   ```

> If `dlib` fails to build on Windows, download a matching wheel from
> [https://www.lfd.uci.edu/\~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and install with:
>
> ```bash
> pip install path\to\dlib-19.xx.x-cpxx-cpxx-win_amd64.whl
> ```

---

##  Project Structure

```
face_recognition_project/
├── known_faces/               # Subfolders named after each person
│   ├── AhmedouYahye/          # Images of the person named "AhmedouYahye"/               
│   └── friend2/
├── models/                    # Cached face embeddings
│   └── encodings.pkl
├── logs/                      # Recognition logs
│   └── recognition.log
├── train.py                   # Encodes known faces, saves embeddings
├── app.py                     # Real-time capture, recognition & logging
├── requirements.txt           # pip install -r requirements.txt
└── README.md                  # Project overview & usage
```

---

##  Usage

### 1. Train Encodings

```bash
python train.py
```

* **Output**:

  ```
  [OK] Encodings saved to models/encodings.pkl
  ```

### 2. Run Real-time Recognition

```bash
python app.py
```

* **Press `q`** to quit.
* **Check** `logs/recognition.log` for entries like:

  ```
  2025-05-10 18:32:28 - AhmedouYahye recognized
  ```

---

##  Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Open a Pull Request

---

##  License

This project is licensed under the **MIT License**.

---

##  References

* **face\_recognition** GitHub: The world’s simplest facial recognition API for Python and the command line
* **opencv-python** PyPI: Pre-built OpenCV packages for Python
* **Pillow** PyPI: The friendly PIL fork for image processing
* **NumPy** PyPI: Fundamental package for scientific computing in Python
