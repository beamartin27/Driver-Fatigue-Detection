"""Synthetic fixtures so tests don't depend on the real captured dataset."""
import numpy as np
import cv2
import pytest


@pytest.fixture
def fake_dataset(tmp_path):
    """Build a tiny on-disk dataset with 3 fake videos x 4 frames each.

    Layout mirrors the real one:
        frames/{label}/{video}_f{idx:05d}.jpg
        landmarks/{label}/{video}_f{idx:05d}_px.npy   (478, 2) int32
    """
    frames_root = tmp_path / "frames"
    lm_root     = tmp_path / "landmarks"

    rng = np.random.default_rng(0)
    rows = []
    for label, label_int in [("trigger", 1), ("non_trigger", 0)]:
        for v_idx in range(3):
            video = f"VID{v_idx}_{label}"
            for f_idx in range(4):
                stem = f"{video}_f{f_idx:05d}"
                img = (rng.integers(0, 255, (240, 320, 3))).astype(np.uint8)
                (frames_root / label).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frames_root / label / f"{stem}.jpg"), img)

                lm = rng.integers(60, 260, (478, 2)).astype(np.int32)
                (lm_root / label).mkdir(parents=True, exist_ok=True)
                np.save(lm_root / label / f"{stem}_px.npy", lm)
                np.save(lm_root / label / f"{stem}.npy", (lm / 320.0).astype(np.float32))

                rows.append({
                    "stem": stem, "video": video, "label": label,
                    "label_int": label_int, "frame_idx": f_idx,
                })

    return {"frames_root": frames_root, "lm_root": lm_root, "rows": rows}
