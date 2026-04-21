# Gesture Activation Baseline (OK -> Peace)

This module implements a baseline activation logic for Driver Fatigue Detection:

- Output is `inactive` by default.
- Output becomes `activated` only when the required ordered sequence is completed in time.
- Required sequence (default): `ok` then `peace`.
- If gesture order is wrong, timeout is exceeded, or gestures are not recognized: stay `inactive`.

## Pipeline

1. Load frames from either:
   - videos, or
   - frame folders.
2. Run MediaPipe Hands to extract 21 hand landmarks.
3. Apply rule-based gesture recognition (`peace`, `ok`, `unknown`).
4. Feed predicted gestures into an ordered sequence FSM.
5. Enforce timeout, debounce, and consecutive-frame gesture confirmation.
6. Emit final status per frame (`inactive`/`activated`).
7. Fire one-shot activation trigger on the first `activated` frame.

## Files

- `config.py`: tunable thresholds and timeout configuration.
- `io_utils.py`: dataset sample discovery and frame iteration.
- `landmarks.py`: MediaPipe hand landmark extraction.
- `gesture_recognizer.py`: rule-based peace/ok detector.
- `sequence_fsm.py`: ordered sequence + timeout state machine.
- `activation_engine.py`: high-level frame processing pipeline.
- `activation_trigger.py`: one-shot activation event hook.
- `run_activation.py`: CLI runner for dataset mode and webcam mode.

## Customization Notes

The baseline rules in `gesture_recognizer.py` are placeholders for robust gesture definitions and should be tuned with your recorded in-vehicle data.

Suggested tuning targets:

- `ok_thumb_index_dist_max`
- `finger_extended_margin`
- `peace_ring_bent_margin`
- `peace_pinky_bent_margin`

## Run

From the project root:

```bash
source venvCompVision/bin/activate
pip install -r src/gesture_activation/requirements_gesture_activation.txt

# Evaluate on your dataset folders
echo "Dataset mode"
python -m src.gesture_activation.run_activation \
  --mode dataset \
  --data-root data/activation \
  --timeout-s 3.0

# Run real-time webcam demo
echo "Webcam mode"
python -m src.gesture_activation.run_activation \
  --mode webcam \
  --timeout-s 1.3 \
  --camera-id 0
```
