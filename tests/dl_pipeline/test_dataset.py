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
