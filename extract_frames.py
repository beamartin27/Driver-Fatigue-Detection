import cv2
import os
import argparse
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_EVERY_N_FRAMES = 4      # keep 1 out of every 4 frames (~4fps from 15fps source)
IMG_EXTENSION         = ".jpg"
JPEG_QUALITY          = 90
# ────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: Path, output_dir: Path, sample_every: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Could not open {video_path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx   = 0
    saved_count = 0
    stem        = video_path.stem   # filename without extension

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            out_name = output_dir / f"{stem}_f{frame_idx:05d}{IMG_EXTENSION}"
            cv2.imwrite(str(out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved_count += 1
        frame_idx += 1

    cap.release()
    return saved_count


def process_split(input_dir: Path, output_dir: Path, label: str, sample_every: int):
    video_files = sorted(
        [f for f in input_dir.iterdir()
         if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    )
    if not video_files:
        print(f"  [WARN] No video files found in {input_dir}")
        return

    print(f"\n── {label.upper()} ({len(video_files)} videos) ──")
    total_frames = 0
    for vf in video_files:
        n = extract_frames(vf, output_dir / label, sample_every)
        print(f"  {vf.name:45s}  →  {n:4d} frames")
        total_frames += n
    print(f"  Total saved: {total_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from fatigue detection dataset")
    parser.add_argument("--input",        type=str, default="data/raw_videos",
                        help="Root folder with trigger/ and non_trigger/ subfolders")
    parser.add_argument("--output",       type=str, default="data/frames",
                        help="Output folder (will create trigger/ and non_trigger/ inside)")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY_N_FRAMES,
                        help="Keep 1 frame every N frames (default: 4)")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)

    trigger_in     = input_root / "trigger"
    non_trigger_in = input_root / "non_trigger"

    for p in [trigger_in, non_trigger_in]:
        if not p.exists():
            print(f"[ERROR] Expected folder not found: {p}")
            print("  Make sure your raw videos are organised as:")
            print("    data/raw_videos/trigger/")
            print("    data/raw_videos/non_trigger/")
            return

    process_split(trigger_in,     output_root, "trigger",     args.sample_every)
    process_split(non_trigger_in, output_root, "non_trigger", args.sample_every)

    print(f"\nDone. Frames saved to: {output_root.resolve()}")
    print("You can now commit the data/frames/ folder to Git.")


if __name__ == "__main__":
    main()