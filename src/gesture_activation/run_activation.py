import argparse
from pathlib import Path

import cv2

from src.gesture_activation.activation_engine import ActivationEngine
from src.gesture_activation.activation_trigger import ActivationTrigger
from src.gesture_activation.config import ActivationConfig
from src.gesture_activation.io_utils import iter_class_samples, iter_sample_frames
from src.gesture_activation.types import SystemStatus


DEFAULT_CLASSES = ["correct", "incorrect_order", "incorrect_time", "idle"]


def resolve_data_root(input_root: Path) -> Path:
    if input_root.exists():
        return input_root

    # Fallback for existing dataset folder typo: "actvation".
    alt = input_root.parent / "actvation"
    if input_root.name == "activation" and alt.exists():
        print(f"[WARN] '{input_root}' not found. Using '{alt}' instead.")
        return alt

    return input_root


def run_dataset_mode(data_root: Path, timeout_s: float, assumed_fps: float, classes: list[str]):
    data_root = resolve_data_root(data_root)

    cfg = ActivationConfig(sequence_timeout_s=timeout_s)
    engine = ActivationEngine(cfg)

    print("=" * 72)
    print("Gesture Activation Dataset Evaluation")
    print(f"Data root: {data_root}")
    print(f"Classes : {classes}")
    print(f"Timeout : {timeout_s:.2f}s")
    print("=" * 72)

    per_class_totals = {c: {"samples": 0, "activated": 0} for c in classes}
    trigger = ActivationTrigger()

    try:
        for class_name, sample_path in iter_class_samples(data_root, classes):
            per_class_totals[class_name]["samples"] += 1
            engine.reset()
            trigger.reset()

            last_result = None
            for frame_idx, ts, frame in iter_sample_frames(sample_path, assumed_fps=assumed_fps):
                last_result = engine.process_frame(frame, timestamp_s=ts, frame_index=frame_idx)
                trigger.maybe_fire(last_result)
                if last_result.status == SystemStatus.ACTIVATED:
                    break

            activated = last_result is not None and last_result.status == SystemStatus.ACTIVATED
            if activated:
                per_class_totals[class_name]["activated"] += 1

            final_status = "activated" if activated else "inactive"
            print(f"[{class_name:14s}] {sample_path.name:30s} -> {final_status}")

    finally:
        engine.close()

    print("\nSummary")
    print("-" * 72)
    for class_name in classes:
        total = per_class_totals[class_name]["samples"]
        active = per_class_totals[class_name]["activated"]
        rate = 0.0 if total == 0 else (100.0 * active / total)
        print(f"{class_name:14s}: activated {active:3d} / {total:3d} ({rate:6.2f}%)")


def run_webcam_mode(timeout_s: float, camera_id: int = 0):
    cfg = ActivationConfig(sequence_timeout_s=timeout_s)
    engine = ActivationEngine(cfg)
    trigger = ActivationTrigger(
        callback=lambda r: print(
            f"[TRIGGER] Activation fired at t={r.timestamp_s:.2f}s (frame={r.frame_index})."
        )
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}")
        engine.close()
        return

    print("Webcam mode running. Press 'q' to quit, 'r' to reset sequence.")
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = frame_idx / fps
            result = engine.process_frame(frame, timestamp_s=ts, frame_index=frame_idx)
            trigger.maybe_fire(result)

            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Status: {result.status.value}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if result.status == SystemStatus.ACTIVATED else (0, 0, 255),
                2,
            )
            cv2.putText(
                overlay,
                f"Gesture: {result.gesture.value}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                f"Progress: {result.sequence_progress}/{len(cfg.target_sequence)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Gesture Activation", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                engine.reset()
                trigger.reset()

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-based activation baseline")
    parser.add_argument(
        "--mode",
        choices=["dataset", "webcam"],
        default="dataset",
        help="Run on activation dataset or webcam.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/activation",
        help="Root folder containing class folders (correct, incorrect_order, incorrect_time, idle).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3.0,
        help="Maximum allowed duration to complete the gesture sequence.",
    )
    parser.add_argument(
        "--assumed-fps",
        type=float,
        default=15.0,
        help="FPS assumption when sample input is an image sequence.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Dataset class folders to evaluate.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID for webcam mode.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "dataset":
        run_dataset_mode(
            data_root=Path(args.data_root),
            timeout_s=args.timeout_s,
            assumed_fps=args.assumed_fps,
            classes=args.classes,
        )
    else:
        run_webcam_mode(timeout_s=args.timeout_s, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
