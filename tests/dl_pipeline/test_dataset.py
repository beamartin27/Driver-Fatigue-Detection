import numpy as np
import pytest
from src.dl_pipeline.dataset import build_index, split_indices


def test_build_index_pairs_frames_and_landmarks(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    assert len(df) == 24                                      # 3 vids x 4 frames x 2 labels
    assert set(df.columns) >= {"frame_path", "landmarks_path", "label", "video"}
    assert df["frame_path"].apply(lambda p: p.exists()).all()
    assert df["landmarks_path"].apply(lambda p: p.exists()).all()
    assert set(df["label"].unique()) == {0, 1}


def test_build_index_extracts_video_id_from_stem(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    assert (df.groupby("video").size() == 4).all()


def test_split_indices_groups_by_video_with_seed_42(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    train_idx, test_idx = split_indices(df, test_size=0.25, random_state=42)
    train_videos = set(df.iloc[train_idx]["video"])
    test_videos  = set(df.iloc[test_idx]["video"])
    assert train_videos.isdisjoint(test_videos)
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_split_indices_is_deterministic(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    a = split_indices(df, random_state=42)
    b = split_indices(df, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


import torch
from src.dl_pipeline.dataset import face_crop_from_landmarks, FrameDataset


def test_face_crop_returns_square_bgr_within_bounds():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 200:500] = 200
    lm = np.zeros((478, 2), dtype=np.int32)
    lm[:468] = np.random.default_rng(0).integers(low=[200, 100], high=[500, 300], size=(468, 2))
    crop = face_crop_from_landmarks(img, lm, out_size=224)
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_face_crop_handles_degenerate_landmarks():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = np.zeros((478, 2), dtype=np.int32)
    crop = face_crop_from_landmarks(img, lm, out_size=224)
    assert crop is not None
    assert crop.shape == (224, 224, 3)


def test_frame_dataset_yields_tensor_and_label(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    ds = FrameDataset(df, train=False)
    sample = ds[0]
    assert "image" in sample and "label" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["label"] in (0, 1)


from src.dl_pipeline.dataset import build_windows, WindowDataset


def test_build_windows_emits_correct_shape(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"])
    windows = build_windows(df, window=3, stride=1)
    assert all(len(w["frame_indices"]) == 3 for w in windows)
    for w in windows:
        last_label = df.iloc[w["frame_indices"][-1]]["label"]
        assert w["label"] == last_label


def test_build_windows_does_not_cross_videos(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    windows = build_windows(df, window=3, stride=1)
    for w in windows:
        videos = df.iloc[w["frame_indices"]]["video"].unique()
        assert len(videos) == 1, "Window must not span multiple videos"


def test_window_dataset_returns_feature_sequence_when_features_provided(fake_dataset):
    df = build_index(fake_dataset["frames_root"], fake_dataset["lm_root"]).sort_values(
        ["video", "frame_path"]
    ).reset_index(drop=True)
    fake_features = np.random.default_rng(0).standard_normal((len(df), 576)).astype(np.float32)
    windows = build_windows(df, window=3, stride=1)
    ds = WindowDataset(df, windows, features=fake_features)
    sample = ds[0]
    assert sample["features"].shape == (3, 576)
    assert sample["features"].dtype == torch.float32
    assert sample["label"] in (0, 1)
