from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_media_samples(class_dir: Path) -> List[Path]:
    """Return sample paths under a class folder.

    Supports two layouts:
    - class_dir/<video files>
    - class_dir/<sample_dir>/<frame files>
    """
    if not class_dir.exists():
        return []

    children = sorted(p for p in class_dir.iterdir() if not p.name.startswith("."))
    media_files = [p for p in children if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    sample_dirs = [p for p in children if p.is_dir()]

    if media_files:
        return media_files
    if sample_dirs:
        return sample_dirs
    return []


def iter_frames_from_video(video_path: Path) -> Generator[Tuple[int, float, any], None, None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        yield frame_idx, ts, frame
        frame_idx += 1

    cap.release()


def iter_frames_from_folder(folder_path: Path, assumed_fps: float = 15.0) -> Generator[Tuple[int, float, any], None, None]:
    image_files = sorted(
        p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    for frame_idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        ts = frame_idx / assumed_fps
        yield frame_idx, ts, frame


def iter_sample_frames(sample_path: Path, assumed_fps: float = 15.0):
    if sample_path.is_file() and sample_path.suffix.lower() in VIDEO_EXTS:
        return iter_frames_from_video(sample_path)
    if sample_path.is_dir():
        return iter_frames_from_folder(sample_path, assumed_fps=assumed_fps)
    return iter(())


def iter_class_samples(data_root: Path, class_names: Iterable[str]):
    for class_name in class_names:
        class_dir = data_root / class_name
        for sample in find_media_samples(class_dir):
            yield class_name, sample
