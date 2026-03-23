"""
landmark_extractor.py
=====================
Member 2 — Face detection + landmark extraction pipeline.

Responsibilities:
  - Detect face in each frame using MediaPipe FaceLandmarker (Tasks API)
  - Extract all 478 landmarks (468 face + 10 iris with refine_landmarks=True)
  - Expose landmark subsets needed by Member 3 (EAR/MAR) and Member 4 (DL)
  - Process full dataset folders and export to CSV / numpy

Setup (run once):
  pip install mediapipe opencv-python
  python landmark_extractor.py --download-model   # downloads face_landmarker.task

Usage:
  # Process the full dataset
  python landmark_extractor.py --frames data/frames --output data/landmarks --confidence 0.3

  # Quick sanity check on a single image
  python landmark_extractor.py --image path/to/frame.jpg
"""

import re
import os
import csv
import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe Tasks API (mediapipe >= 0.10.13) ──────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── CONSTANTS ────────────────────────────────────────────────────────────────

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path("models/face_landmarker.task")

# Approximate ms per frame — matches the ~15fps source videos
MS_PER_FRAME = 66

# EAR — 6 landmarks per eye (P1..P6 going clockwise around the eye contour)
# Formula: EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# MAR — 8 outer lip landmarks
# Formula: MAR = (|P2-P8|+|P3-P7|+|P4-P6|) / (2 * |P1-P5|)
MOUTH_MAR = [61, 39, 0, 269, 291, 405, 17, 181]

# Head pose — 6 3D reference points (nose tip, chin, eye corners, mouth corners)
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Full eye regions (for DL cropping / visualisation)
LEFT_EYE_REGION  = [362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_REGION = [33,  7,   163, 144, 145, 153, 154, 155,
                    133, 173, 157, 158, 159, 160, 161, 246]

# ── MODEL DOWNLOAD ───────────────────────────────────────────────────────────

def download_model(model_path: Path = MODEL_PATH) -> None:
    """Download the MediaPipe face landmarker model if not present."""
    if model_path.exists() and model_path.stat().st_size > 0:
        print(f"Model already present: {model_path}")
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from:\n  {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Saved to {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")

# ── HELPERS ──────────────────────────────────────────────────────────────────

def _source_video_name(img_path: Path) -> str:
    """Extract source video stem from frame filename, e.g. WIN_..._Pro_f00012 -> WIN_..._Pro."""
    m = re.match(r"(.+)_f\d+", img_path.stem)
    return m.group(1) if m else img_path.stem

# ── CORE EXTRACTOR ───────────────────────────────────────────────────────────

class LandmarkExtractor:
    """
    Wraps MediaPipe FaceLandmarker in VIDEO mode for temporal tracking.

    VIDEO mode keeps tracking the face across frames even through brief
    occlusions or extreme angles, which significantly improves detection
    rate on a frame-extracted dataset compared to IMAGE mode.

    Important: call reset_tracker() at the start of each new source video
    so timestamps restart and the tracker does not carry state across clips.

    Parameters
    ----------
    model_path               : path to face_landmarker.task
    min_detection_confidence : threshold for face detection (default 0.3)
    """

    def __init__(self,
                 model_path: Path = MODEL_PATH,
                 min_detection_confidence: float = 0.3):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Run:  python landmark_extractor.py --download-model"
            )

        self._confidence = min_detection_confidence
        self._model_path = model_path
        self._timestamp_ms = 0
        self._landmarker = self._build_landmarker()

    def _build_landmarker(self):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self._confidence,
            min_face_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def reset_tracker(self):
        """
        Reset timestamp and rebuild the landmarker.
        Call this at the start of every new source video.
        """
        self._landmarker.close()
        self._timestamp_ms = 0
        self._landmarker = self._build_landmarker()

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def process_frame(self, bgr_image: np.ndarray) -> dict | None:
        """
        Run landmark detection on a single BGR frame (as returned by cv2).

        Returns a dict with keys:
          'landmarks_norm'  : np.ndarray (478, 3)  -- normalised (x, y, z) in [0,1]
          'landmarks_px'    : np.ndarray (478, 2)  -- pixel (u, v) coordinates
          'image_shape'     : (height, width)

        Returns None if no face is detected.
        """
        h, w = bgr_image.shape[:2]
        rgb    = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, self._timestamp_ms)
        self._timestamp_ms += MS_PER_FRAME

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]   # first (only) face

        # Normalised coordinates
        norm = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        # Pixel coordinates (u, v) — z is depth, not a pixel coord
        px = np.stack([norm[:, 0] * w, norm[:, 1] * h], axis=1).astype(np.int32)

        return {
            "landmarks_norm": norm,          # shape (478, 3)
            "landmarks_px":   px,            # shape (478, 2)
            "image_shape":    (h, w),
        }

    def process_image_file(self, path: str | Path) -> dict | None:
        """Convenience wrapper: load a JPEG/PNG and call process_frame."""
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Could not read image: {path}")
        result = self.process_frame(img)
        if result is not None:
            result["source_path"] = str(path)
        return result

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# ── LANDMARK SUBSET HELPERS ──────────────────────────────────────────────────
# These are the functions Member 3 will call directly.

def get_ear_landmarks(landmarks_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 6 EAR landmark points for each eye as pixel coordinates.

    Returns
    -------
    left_eye  : (6, 2) array  -- ordered P1..P6 for EAR formula
    right_eye : (6, 2) array  -- ordered P1..P6 for EAR formula
    """
    left  = landmarks_px[LEFT_EYE_EAR]
    right = landmarks_px[RIGHT_EYE_EAR]
    return left.astype(np.float32), right.astype(np.float32)


def get_mar_landmarks(landmarks_px: np.ndarray) -> np.ndarray:
    """
    Return the 8 MAR landmark points as pixel coordinates.

    Returns
    -------
    mouth : (8, 2) array -- ordered P1..P8 for MAR formula
    """
    return landmarks_px[MOUTH_MAR].astype(np.float32)


def get_head_pose_landmarks(landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Return 6 normalised landmarks used for head pose estimation.

    Returns
    -------
    points : (6, 3) array -- (x, y, z) normalised coordinates
    """
    return landmarks_norm[HEAD_POSE_LANDMARKS].astype(np.float32)


def get_eye_region_landmarks(landmarks_px: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return full eye region contour points (for DL ROI cropping).

    Returns
    -------
    left_region  : (16, 2) pixel array
    right_region : (16, 2) pixel array
    """
    left  = landmarks_px[LEFT_EYE_REGION]
    right = landmarks_px[RIGHT_EYE_REGION]
    return left.astype(np.float32), right.astype(np.float32)

# ── DATASET EXPORT ───────────────────────────────────────────────────────────

def process_dataset(frames_root: Path,
                    output_root: Path,
                    model_path: Path = MODEL_PATH,
                    min_confidence: float = 0.3) -> None:
    """
    Process all frames in data/frames/{trigger, non_trigger}/
    and save landmark arrays to data/landmarks/{trigger, non_trigger}/.

    Output per frame (if face detected):
      <output_root>/<label>/<stem>.npy    -- full (478, 3) normalised landmarks
      <output_root>/<label>/<stem>_px.npy -- full (478, 2) pixel landmarks

    Also writes a summary CSV: <output_root>/dataset_summary.csv
    with columns: path, label, detected, num_landmarks
    """
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    with LandmarkExtractor(model_path, min_confidence) as extractor:
        for label in ["trigger", "non_trigger"]:
            in_dir  = frames_root / label
            out_dir = output_root / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"[WARN] Folder not found, skipping: {in_dir}")
                continue

            image_files = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
            detected = 0
            missed   = 0
            current_source = None

            print(f"\n-- {label.upper()} ({len(image_files)} frames) --")

            for img_path in image_files:
                # Reset tracker at the boundary between source videos
                source = _source_video_name(img_path)
                if source != current_source:
                    extractor.reset_tracker()
                    current_source = source

                result = extractor.process_image_file(img_path)
                stem   = img_path.stem

                if result is not None:
                    np.save(out_dir / f"{stem}.npy",    result["landmarks_norm"])
                    np.save(out_dir / f"{stem}_px.npy", result["landmarks_px"])
                    summary_rows.append([str(img_path), label, True,  478])
                    detected += 1
                else:
                    summary_rows.append([str(img_path), label, False, 0])
                    missed += 1

            total = detected + missed
            pct   = 100 * detected / total if total > 0 else 0
            print(f"  Detected: {detected}/{total}  ({pct:.1f}%)")
            if missed > 0:
                print(f"  [INFO] {missed} undetected frames -- likely extreme head angles "
                      "or heavy occlusion. These are excluded from training automatically.")

    # Write summary CSV
    csv_path = output_root / "dataset_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "detected", "num_landmarks"])
        writer.writerows(summary_rows)

    print(f"\nSummary CSV saved to: {csv_path}")
    print(f"Landmark arrays saved to: {output_root}")

# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Landmark extraction pipeline")
    parser.add_argument("--download-model", action="store_true",
                        help="Download the MediaPipe face landmarker model and exit")
    parser.add_argument("--frames",  type=str, default="data/frames",
                        help="Root folder with trigger/ and non_trigger/ subfolders")
    parser.add_argument("--output",  type=str, default="data/landmarks",
                        help="Output folder for .npy landmark files")
    parser.add_argument("--model",   type=str, default=str(MODEL_PATH),
                        help="Path to face_landmarker.task")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Minimum face detection confidence (default: 0.3)")
    parser.add_argument("--image",   type=str, default=None,
                        help="Run on a single image and print landmark info")
    args = parser.parse_args()

    model_path = Path(args.model)

    if args.download_model:
        download_model(model_path)
        return

    if args.image:
        with LandmarkExtractor(model_path, args.confidence) as extractor:
            result = extractor.process_image_file(args.image)
        if result is None:
            print("No face detected in image.")
        else:
            lm_px   = result["landmarks_px"]
            lm_norm = result["landmarks_norm"]
            print(f"Detected {len(lm_norm)} landmarks in {args.image}")
            l_eye, r_eye = get_ear_landmarks(lm_px)
            mouth        = get_mar_landmarks(lm_px)
            print(f"Left EAR points (px):  {l_eye}")
            print(f"Right EAR points (px): {r_eye}")
            print(f"Mouth MAR points (px): {mouth}")
        return

    # Full dataset mode
    process_dataset(
        frames_root=Path(args.frames),
        output_root=Path(args.output),
        model_path=model_path,
        min_confidence=args.confidence,
    )

if __name__ == "__main__":
    main()