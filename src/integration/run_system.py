"""Member 5 - Unified real-time pipeline: gesture -> landmarks -> fusion -> HUD."""

from __future__ import annotations

import argparse
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.dl_pipeline import config as dl_config
from src.dl_pipeline.predict import DLPredictor
from src.gesture_activation.activation_engine import ActivationEngine
from src.gesture_activation.config import ActivationConfig
from src.gesture_activation.types import SystemStatus
from src.integration.classical_infer import ClassicalPredictor
from src.integration.fusion import FusionConfig, fuse_predictions
from src.integration.overlay import draw_activation_overlay, draw_monitoring_overlay

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_existing(path_str: str, patterns: tuple[str, ...]) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    rel_repo = REPO_ROOT / candidate
    if rel_repo.exists():
        return rel_repo.resolve()
    for pat in patterns:
        rel = REPO_ROOT / pat
        if "*" not in pat and "**" not in pat:
            if rel.exists():
                return rel.resolve()
            continue
        matches = sorted(glob.glob(str(rel)))
        if matches:
            return Path(matches[0])
    return candidate.resolve()


class FrameClock:
    def __init__(self, fps: float):
        self.fps = max(fps, 1e-3)
        self.frame_idx = 0

    def tick(self) -> float:
        t = self.frame_idx / self.fps
        self.frame_idx += 1
        return t


class LSTMRollingBuffer:
    def __init__(self, maxlen: int = dl_config.WINDOW_SIZE):
        self.frames = deque(maxlen=maxlen)
        self.landmarks = deque(maxlen=maxlen)

    def push(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray):
        self.frames.append(frame_bgr.copy())
        self.landmarks.append(np.asarray(landmarks_px).copy())

    def full(self) -> bool:
        return len(self.frames) == self.frames.maxlen and self.frames.maxlen is not None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated fatigue system: gesture gate + classical + DL + fusion HUD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional MP4/MOV/etc. instead of webcam (still shows GUI unless --headless).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=4.5,
        help="Gesture sequence timeout (seconds window for OK then Peace).",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.3,
        help="MediaPipe FaceLandmarker detection confidence.",
    )
    parser.add_argument(
        "--classical-model-path",
        type=str,
        default=str(REPO_ROOT / "outputs" / "classical_pipeline" / "svm.joblib"),
        help="Member 3 joblib path (svm/random_forest/knn).",
    )
    parser.add_argument(
        "--cnn-ckpt",
        type=str,
        default=str(REPO_ROOT / "outputs" / "dl_pipeline" / "cnn_mobilenetv3.pt"),
        help="Member 4 CNN checkpoint.",
    )
    parser.add_argument(
        "--lstm-ckpt",
        type=str,
        default=str(REPO_ROOT / "outputs" / "dl_pipeline" / "cnn_lstm.pt"),
        help="CNN-LSTM checkpoint (optional; omit with --skip-lstm).",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip CNN-LSTM (lower-latency path).",
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Deep branch off (until CNN ckpt exists). Classical-only fusion.",
    )
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="weighted_mean",
        choices=["mean", "max", "min", "weighted_mean", "both"],
    )
    parser.add_argument("--fusion-threshold", type=float, default=0.5)
    parser.add_argument("--w-classical", type=float, default=0.45)
    parser.add_argument("--w-cnn", type=float, default=0.45)
    parser.add_argument("--w-lstm", type=float, default=0.10)
    parser.add_argument(
        "--reset-face-tracker-on-arm",
        action="store_true",
        help="Rebuild FaceLandmarker when switching from gestures to monitoring.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device tag (cuda, cpu).")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No GUI; print fused alerts instead.",
    )
    parser.add_argument(
        "--arm-on-start",
        action="store_true",
        help="Skip gesture gate (debug / automated runs).",
    )
    return parser.parse_args()


def _open_capture(args) -> tuple[cv2.VideoCapture, float]:
    if args.video:
        cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return cap, fps if fps > 0 else 30.0
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, fps if fps > 0 else 30.0


def run():
    args = parse_args()

    import sys

    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from landmark_extractor import LandmarkExtractor  # pylint: disable=import-outside-toplevel

    fusion_cfg = FusionConfig(
        strategy=args.fusion_strategy,
        threshold=args.fusion_threshold,
        classical_weight=args.w_classical,
        cnn_weight=args.w_cnn,
        lstm_weight=args.w_lstm,
    )

    classical_path = _resolve_existing(
        args.classical_model_path,
        (
            "outputs/classical_pipeline/svm.joblib",
            "outputs/classical_pipeline/random_forest.joblib",
            "outputs/classical_pipeline/knn.joblib",
            "outputs/classical_pipeline/*.joblib",
        ),
    )

    gesture_cfg = ActivationConfig(sequence_timeout_s=args.timeout_s)
    gest_engine = ActivationEngine(gesture_cfg)
    extractor = LandmarkExtractor(min_detection_confidence=args.face_confidence)

    classical = ClassicalPredictor(classical_path) if classical_path.exists() else None

    predictor: DLPredictor | None = None
    ckpt_used = "off"
    if not args.skip_dl:
        cnn_ckpt = Path(_resolve_existing(args.cnn_ckpt, ("outputs/dl_pipeline/cnn_mobilenetv3.pt",)))
        if not cnn_ckpt.exists():
            print(
                f"[WARN] CNN checkpoint missing at {cnn_ckpt}.\n"
                "Train: python -m src.dl_pipeline.run_dl_pipeline train_cnn ..."
            )
        else:
            lstm_ckpt_arg: Path | None = Path(_resolve_existing(args.lstm_ckpt, ("outputs/dl_pipeline/cnn_lstm.pt",)))
            if args.skip_lstm or not lstm_ckpt_arg.exists():
                lstm_ckpt_arg = None
                if not args.skip_lstm:
                    print("[INFO] CNN-LSTM checkpoint absent - fusion uses CNN (+ classical) only.")
            predictor = DLPredictor(
                cnn_ckpt=cnn_ckpt,
                lstm_ckpt=lstm_ckpt_arg,
                device=args.device,
            )
            ckpt_used = "cnn+lstm" if lstm_ckpt_arg is not None else "cnn_only"

    if classical is None and predictor is None:
        raise RuntimeError(
            "No runnable branch.\n"
            f"- Classical model path: {classical_path} ({'MISSING' if not classical_path.exists() else 'found'}).\n"
            f"- DL: {'skipped via --skip-dl' if args.skip_dl else 'cnn checkpoint unavailable'}.\n"
            "Train baseline models before running integration."
        )

    cap, native_fps = _open_capture(args)
    clk = FrameClock(native_fps)
    lstm_buf = LSTMRollingBuffer(dl_config.WINDOW_SIZE)

    phase = "gesture_wait" if not args.arm_on_start else "monitoring"

    fps_ema = 30.0
    t_wall_prev = perf_counter()

    strat_label = getattr(fusion_cfg.strategy, "value", str(fusion_cfg.strategy))
    print(
        "Integrated driver-fatigue pipeline\n"
        f" phase start    : {phase}\n"
        f" classical      : {classical_path.name if classical else 'DISABLED'}\n"
        f" deep           : {ckpt_used}\n"
        f" fusion         : {strat_label} @ threshold {fusion_cfg.threshold}\n"
        "----------------------------------------------------------------\n"
        " Keys: q quit | r reset sequence + face tracker | x toggle gesture bypass\n"
        "================================================================"
    )

    fused_last = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            ts = clk.tick()

            tw = perf_counter()
            fps_inst = 1.0 / max(tw - t_wall_prev, 1e-6)
            t_wall_prev = tw
            fps_ema = 0.92 * fps_ema + 0.08 * fps_inst

            if phase == "gesture_wait":
                gres = gest_engine.process_frame(frame, timestamp_s=ts, frame_index=int(clk.frame_idx))
                canvas = draw_activation_overlay(frame, gres)
                if gres.status == SystemStatus.ACTIVATED:
                    phase = "monitoring"
                    if args.reset_face_tracker_on_arm:
                        extractor.reset_tracker()
                    lstm_buf.frames.clear()
                    lstm_buf.landmarks.clear()
                    print("[STATE] Gesture OK - monitoring ENABLED.")

            elif phase == "monitoring":
                landmarks = extractor.process_frame(frame)

                classical_out = None
                dl_out_cnn = None
                dl_out_ls = None

                no_face_banner = landmarks is None
                lm_px = None if no_face_banner else landmarks["landmarks_px"]

                if not no_face_banner and lm_px is not None:
                    futures = []

                    def submit(pool: ThreadPoolExecutor):
                        nonlocal futures
                        if classical is not None:
                            futures.append(pool.submit(classical.predict, frame, lm_px))
                        if predictor is not None:
                            futures.append(pool.submit(predictor.predict_frame, frame, lm_px))

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        submit(pool)
                        results = []
                        for fut in futures:
                            try:
                                results.append(fut.result())
                            except Exception as exc:  # noqa: BLE001
                                results.append({"proba": 0.0, "label": 0, "ok": False, "detail": repr(exc)})
                        idx = 0
                        if classical is not None:
                            classical_out = results[idx]
                            idx += 1
                        else:
                            classical_out = None
                        if predictor is not None:
                            dl_out_cnn = results[idx]
                        lstm_buf.push(frame, lm_px)
                        dl_out_ls = None
                        if predictor is not None and predictor.lstm is not None and lstm_buf.full():
                            try:
                                dl_out_ls = predictor.predict_window(
                                    list(lstm_buf.frames),
                                    list(lstm_buf.landmarks),
                                )
                                dl_out_ls["ok"] = True
                            except Exception as exc:  # noqa: BLE001
                                dl_out_ls = {"proba": 0.0, "label": 0, "ok": False, "detail": repr(exc)}

                    fused_last = fuse_predictions(classical_out, dl_out_cnn, dl_out_ls, fusion_cfg)
                    canvas = draw_monitoring_overlay(
                        frame,
                        fusion=fused_last,
                        fps=fps_ema,
                        headless_note="(no GUI)" if args.headless else "",
                    )

                    if args.headless and fused_last.alert:
                        print(f"[ALERT] t={ts:.2f}s fused_proba={fused_last.fused_proba:.3f}")

                else:
                    fused_last = fuse_predictions(
                        {"proba": 0.0, "label": 0, "ok": False, "detail": "no_face"},
                        None,
                        None,
                        fusion_cfg,
                    )
                    canvas = draw_monitoring_overlay(frame, fusion=fused_last, fps=fps_ema, headless_note="no_face")

            else:
                canvas = frame

            if args.headless:
                continue

            cv2.imshow("Driver Fatigue - Integrated", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                gest_engine.reset()
                lstm_buf.frames.clear()
                lstm_buf.landmarks.clear()
                extractor.reset_tracker()
                phase = "gesture_wait" if not args.arm_on_start else "monitoring"
                print("[RESET] Sequence + trackers cleared.")
            if key == ord("x"):
                args.arm_on_start = not args.arm_on_start
                phase = "monitoring" if args.arm_on_start else "gesture_wait"
                print(f"[TOGGLE] bypass gate -> phase={phase}")

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        extractor.close()
        gest_engine.close()


if __name__ == "__main__":
    run()
