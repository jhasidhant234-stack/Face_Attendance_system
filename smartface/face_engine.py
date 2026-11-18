import urllib.request
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        target = MODEL_DIR / filename
        if target.exists() and target.stat().st_size > 100_000:
            continue
        print(f"Downloading model: {filename}")
        request = urllib.request.Request(url, headers={"User-Agent": "SmartFaceAttendance/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, open(target, "wb") as out:
            out.write(response.read())
        if target.stat().st_size < 100_000:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded model appears incomplete: {filename}")

class FaceEngine:
    def __init__(self, score_threshold=0.85):
        ensure_models()
        detector_path = str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
        recognizer_path = str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")
        self.detector = cv2.FaceDetectorYN_create(
            detector_path, "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(recognizer_path, "")
        self.match_threshold = 0.40

    def extract(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        output = []
        if faces is None:
            return output
        for face in faces:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature /= norm
            x, y, bw, bh = [int(v) for v in face[:4]]
            output.append((feature, (x, y, bw, bh), float(face[14])))
        return output

    def match(self, feature, employees):
        best, best_score = None, -1.0
        for employee in employees:
            known = np.asarray(employee["embedding"], dtype=np.float32)
            denom = np.linalg.norm(feature) * np.linalg.norm(known)
            score = float(np.dot(feature, known) / denom) if denom else -1.0
            if score > best_score:
                best, best_score = employee, score
        if best is not None and best_score >= self.match_threshold:
            return best, best_score
        return None, best_score
