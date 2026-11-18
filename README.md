# SmartFace Attendance System

A Windows desktop face-attendance application using Python, Tkinter, OpenCV YuNet/SFace ONNX models and SQLite.

## Features

- Register a person using a unique employee ID and name.
- Capture 10 face samples from the laptop webcam and store an averaged face embedding.
- Recognize registered people from a live webcam stream.
- Mark attendance once per person per calendar day.
- Display dashboard counts and registered employees.
- Delete an employee (and their associated attendance history).
- View attendance by date and export CSV reports.
- Face embeddings are stored in SQLite; raw registration photographs are not saved.

## Requirements

- Windows 10 or 11
- Python 3.10 or 3.11 recommended
- Webcam
- Internet connection for the first run (downloads the two pretrained ONNX models)
- Microsoft Visual C++ runtime may be needed by some Python/OpenCV installations.

## Setup (PowerShell)

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

You can also avoid activation entirely:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## First launch

The app downloads:
- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

They are placed in `models/`. If download fails, check internet access and retry. The app does not download models from an untrusted mirror.

## How to use

1. Open **Register Face**.
2. Enter a unique employee ID and full name.
3. Click **Start face capture (10 samples)**.
4. Keep one person's face visible and slowly vary the head angle until registration completes.
5. Open **Take Attendance** and click **Start attendance camera**.
6. A recognized person is marked once per day. Unknown faces are not recorded.
7. Use **Reports** to select a date and export the report to CSV.

## Project structure

```text
SmartFace_Attendance/
├── app.py
├── requirements.txt
├── README.md
├── smartface/
│   ├── __init__.py
│   ├── database.py
│   └── face_engine.py
├── models/                 # downloaded ONNX files
└── data/                   # SQLite database created at runtime
```

## Recognition threshold

The default cosine similarity threshold is `0.40` in `smartface/face_engine.py`. A higher threshold is stricter and may reject genuine matches; a lower threshold may increase false matches. Calibrate it with several different lighting conditions and camera distances before real use. Do not treat a similarity score as a probability.

## Important limitations and responsible use

- This is a functional starter implementation, not a certified biometric security product.
- It does not include liveness/anti-spoofing. A photograph or screen replay may fool a basic camera-only face matcher. Do not deploy it for payroll, access control, or disciplinary decisions without a tested liveness system and human review.
- Obtain informed consent before collecting biometric data. Restrict access to the database, encrypt backups, define a retention/deletion policy, and comply with applicable Indian privacy and workplace requirements.
- Recognition accuracy varies with camera quality, lighting, pose, and enrollment quality.
- Keep only one person in the frame during registration. The app captures the largest face it detects.
- Attendance is marked once per calendar day; this version does not implement shift schedules, breaks, late rules, or multiple check-in/check-out events.

## Troubleshooting

### `No module named cv2`
Activate the virtual environment and run `pip install -r requirements.txt`.

### Camera cannot open
Close Zoom/Teams/browser camera tabs. Check Windows **Settings → Privacy & security → Camera** and allow desktop apps to access the camera. If you have multiple cameras, edit `cv2.VideoCapture(0, ...)` in `app.py` to `1`.

### `FaceDetectorYN_create` or `FaceRecognizerSF_create` missing
Make sure `opencv-contrib-python` is installed and remove conflicting OpenCV packages:

```powershell
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install -r requirements.txt
```

### Model download fails
Retry with a stable internet connection. The ONNX files must be present in `models/` before the engine can start.

### False matches or missed matches
Improve enrollment lighting, capture clean frontal/angled samples, keep the camera at eye level, and calibrate the threshold. Do not lower it blindly.
