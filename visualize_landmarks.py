"""
visualize_landmarks.py
======================
Member 2 — Sanity check visualization for the landmark pipeline.

Draws detected landmarks on sample frames so the team can visually verify
that EAR/MAR/head-pose points are landing in the right places before
Members 3 and 4 start building on top of them.

Usage:
  # Visualize random samples from the dataset (default: 5 per class)
  python visualize_landmarks.py

  # Visualize a specific image
  python visualize_landmarks.py --image data/frames/trigger/WIN_xxx_f00040.jpg

  # Control how many samples per class
  python visualize_landmarks.py --samples 10

Output:
  Saves annotated images to data/viz/ and opens a preview window.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from landmark_extractor import (
    LandmarkExtractor,
    get_ear_landmarks,
    get_mar_landmarks,
    get_head_pose_landmarks,
    get_eye_region_landmarks,
    MODEL_PATH,
)

# ── DRAWING CONFIG ────────────────────────────────────────────────────────────

COLORS = {
    "left_eye":  (0,   255, 100),   # green
    "right_eye": (0,   180, 255),   # orange
    "mouth":     (255,  80,  80),   # blue-ish
    "all_lm":    (180, 180, 180),   # gray — all 478 points, subtle
    "head_pose": (255, 255,   0),   # yellow
    "label_bg":  (30,   30,  30),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1

# ── DRAWING HELPERS ───────────────────────────────────────────────────────────

def draw_all_landmarks(img: np.ndarray, landmarks_px: np.ndarray) -> None:
    """Draw all 478 landmarks as small gray dots."""
    for (x, y) in landmarks_px:
        cv2.circle(img, (int(x), int(y)), 1, COLORS["all_lm"], -1)


def draw_ear_points(img: np.ndarray, landmarks_px: np.ndarray) -> None:
    """Draw the 6 EAR points for each eye with connecting lines."""
    l_eye, r_eye = get_ear_landmarks(landmarks_px)

    for pts, color, name in [(l_eye, COLORS["left_eye"],  "L"),
                              (r_eye, COLORS["right_eye"], "R")]:
        pts = pts.astype(np.int32)
        # Connect P1-P2-P3-P4 (horizontal) and P2-P6, P3-P5 (vertical)
        cv2.polylines(img, [pts[[0,1,2,3]]], False, color, 1, cv2.LINE_AA)
        cv2.line(img, tuple(pts[1]), tuple(pts[5]), color, 1, cv2.LINE_AA)
        cv2.line(img, tuple(pts[2]), tuple(pts[4]), color, 1, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(img, tuple(pt), 3, color, -1)
            cv2.putText(img, f"P{i+1}", (pt[0]+3, pt[1]-3),
                        FONT, 0.3, color, 1, cv2.LINE_AA)


def draw_mar_points(img: np.ndarray, landmarks_px: np.ndarray) -> None:
    """Draw the 8 MAR points with connecting lines."""
    mouth = get_mar_landmarks(landmarks_px).astype(np.int32)
    color = COLORS["mouth"]
    # Draw outer lip contour
    cv2.polylines(img, [mouth[:4]], False, color, 1, cv2.LINE_AA)
    cv2.polylines(img, [mouth[4:]], False, color, 1, cv2.LINE_AA)
    for i, pt in enumerate(mouth):
        cv2.circle(img, tuple(pt), 3, color, -1)
        cv2.putText(img, f"M{i+1}", (pt[0]+3, pt[1]-3),
                    FONT, 0.3, color, 1, cv2.LINE_AA)


def draw_head_pose_points(img: np.ndarray, landmarks_norm: np.ndarray,
                           image_shape: tuple) -> None:
    """Draw the 6 head-pose reference landmarks."""
    h, w = image_shape
    pts  = get_head_pose_landmarks(landmarks_norm)
    color = COLORS["head_pose"]
    for pt in pts:
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(img, (x, y), 4, color, -1)


def put_label(img: np.ndarray, text: str, pos: tuple,
              color=(255, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = pos
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+2, y+2), COLORS["label_bg"], -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def annotate_frame(bgr: np.ndarray, result: dict, label: str) -> np.ndarray:
    """Draw all landmark groups on a copy of the frame."""
    out = bgr.copy()
    lm_px   = result["landmarks_px"]
    lm_norm = result["landmarks_norm"]
    h, w    = result["image_shape"]

    draw_all_landmarks(out, lm_px)
    draw_ear_points(out, lm_px)
    draw_mar_points(out, lm_px)
    draw_head_pose_points(out, lm_norm, (h, w))

    # Class label + legend
    tag_color = (0, 200, 80) if label == "non_trigger" else (0, 80, 220)
    put_label(out, f"CLASS: {label.upper()}", (10, 20), tag_color)
    put_label(out, "gray=all  green=L-eye  orange=R-eye  blue=mouth  yellow=head-pose",
              (10, h - 10))

    return out

# ── MAIN ──────────────────────────────────────────────────────────────────────

def visualize_single(image_path: str, model_path: Path, confidence: float) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    with LandmarkExtractor(model_path, confidence) as ex:
        result = ex.process_frame(img)

    if result is None:
        print("No face detected.")
        return

    label = "trigger" if "trigger" in image_path and "non_trigger" not in image_path \
            else "non_trigger"
    out = annotate_frame(img, result, label)

    out_path = Path("data/viz") / (Path(image_path).stem + "_viz.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Saved: {out_path}")

    cv2.imshow("Landmark check — press any key", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_samples(frames_root: Path, output_root: Path,
                      model_path: Path, confidence: float,
                      n_samples: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_annotated = []

    with LandmarkExtractor(model_path, confidence) as ex:
        for label in ["trigger", "non_trigger"]:
            in_dir = frames_root / label
            if not in_dir.exists():
                print(f"[WARN] Not found: {in_dir}")
                continue

            files = sorted(in_dir.glob("*.jpg"))
            sample = random.sample(files, min(n_samples, len(files)))
            print(f"\n{label.upper()} — sampling {len(sample)} frames")

            for img_path in sample:
                img    = cv2.imread(str(img_path))
                result = ex.process_frame(img)

                if result is None:
                    print(f"  [SKIP] No face: {img_path.name}")
                    continue

                out      = annotate_frame(img, result, label)
                out_path = output_root / f"{img_path.stem}_viz.jpg"
                cv2.imwrite(str(out_path), out)
                all_annotated.append(out)
                print(f"  Saved: {out_path.name}")

    if not all_annotated:
        print("No annotated frames produced.")
        return

    # Build a contact sheet: 2 columns, auto rows
    cols = 2
    h_thumb, w_thumb = 360, 480
    thumbs = [cv2.resize(f, (w_thumb, h_thumb)) for f in all_annotated]

    # Pad to even number
    if len(thumbs) % cols != 0:
        blank = np.zeros((h_thumb, w_thumb, 3), dtype=np.uint8)
        thumbs.append(blank)

    rows = [np.hstack(thumbs[i:i+cols]) for i in range(0, len(thumbs), cols)]
    sheet = np.vstack(rows)

    sheet_path = output_root / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\nContact sheet saved: {sheet_path}")

    cv2.imshow("Landmark check — press any key to close", sheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize landmark extraction results")
    parser.add_argument("--image",   type=str, default=None,
                        help="Visualize a single image file")
    parser.add_argument("--frames",  type=str, default="data/frames",
                        help="Root folder with trigger/ and non_trigger/")
    parser.add_argument("--output",  type=str, default="data/viz",
                        help="Output folder for annotated images")
    parser.add_argument("--model",   type=str, default=str(MODEL_PATH),
                        help="Path to face_landmarker.task")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of random frames to sample per class (default: 5)")
    args = parser.parse_args()

    if args.image:
        visualize_single(args.image, Path(args.model), args.confidence)
    else:
        visualize_samples(
            frames_root=Path(args.frames),
            output_root=Path(args.output),
            model_path=Path(args.model),
            confidence=args.confidence,
            n_samples=args.samples,
        )

if __name__ == "__main__":
    main()