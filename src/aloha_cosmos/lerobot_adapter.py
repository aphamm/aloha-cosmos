import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def convert_to_bridge_format(
    data_path: str | Path,
    output_path: str | Path,
    camera: str = "observation.images.wrist",
    train_ratio: float = 0.9,
    fps_downsample: int = 10,  # 30fps → 3fps
):
    """
    Convert a LeRobot dataset to Cosmos Bridge format.

    Args:
        data_path: Path to LeRobot dataset
        output_path: Where to write Bridge format dataset
        camera: Which camera to use
        train_ratio: Fraction of episodes for training (rest for validation)
        fps_downsample: Downsample factor for video/state/action (10 = 30fps→3fps)
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    for split in ["train", "val"]:
        (output_path / "annotation" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "videos" / split).mkdir(parents=True, exist_ok=True)

    # Load episode metadata
    episodes = _load_episodes(data_path)
    n_train = int(len(episodes) * train_ratio)

    print(
        f"Total episodes: {len(episodes)}, train: {n_train}, val: {len(episodes) - n_train}"
    )

    # Load tasks from parquet file
    df = pd.read_parquet(data_path / "meta" / "tasks.parquet")
    tasks = {int(row["task_index"]): task_name for task_name, row in df.iterrows()}

    # Process each episode
    for i, ep in enumerate(episodes):
        split = "train" if i < n_train else "val"
        episode_idx = ep["episode_index"]

        task_indices = ep.get("tasks", [0])
        task_text = tasks.get(task_indices[0], f"{task_indices[0]}")

        # Load state/action data for this episode
        states, actions = _load_episode_data(data_path, ep)

        # Create annotation JSON (downsampled to reach target video fps)
        annotation = _create_bridge_annotation(
            episode_idx=episode_idx,
            states=states,
            actions=actions,
            task_text=task_text,
            split=split,
            fps_downsample=fps_downsample,
        )

        # Write annotation
        ann_path = output_path / "annotation" / split / f"{episode_idx}.json"
        with open(ann_path, "w") as f:
            json.dump(annotation, f, indent=2)

        # Extract episode video segment using ffmpeg
        src_video = _get_video_path(data_path, ep, camera)
        dst_video_dir = output_path / "videos" / split / str(episode_idx)
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / "rgb.mp4"

        if src_video.exists():
            # Get episode timestamps
            from_ts = ep[f"videos/{camera}/from_timestamp"]
            to_ts = ep[f"videos/{camera}/to_timestamp"]
            duration = to_ts - from_ts

            # Extract episode segment, downsample to 3fps, encode as mpeg4
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(from_ts),
                "-i",
                str(src_video),
                "-t",
                str(duration),
                "-r",
                "3",  # Downsample to 3fps (from 30fps)
                "-c:v",
                "mpeg4",  # Encode as mpeg4
                "-q:v",
                "5",  # Quality (lower = better, 2-31 range)
                str(dst_video),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        # Save state.npy (downsampled state data)
        state_file = dst_video_dir / "state.npy"
        downsampled_states = states[::fps_downsample]
        np.save(state_file, downsampled_states.astype(np.float64))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(episodes)} episodes")

    print("Conversion complete!")
    print(f"Annotations: {output_path / 'annotation'}")
    print(f"Videos: {output_path / 'videos'}")


def _load_episodes(data_path: Path) -> list[dict]:
    """Load episode metadata from LeRobot parquet files."""
    episodes = []
    episodes_dir = data_path / "meta" / "episodes"

    for parquet_file in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            episodes.append(row.to_dict())

    return sorted(episodes, key=lambda x: x["episode_index"])


def _load_episode_data(
    data_path: Path, episode_info: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Load state and action data for an episode."""
    chunk_idx = episode_info["data/chunk_index"]
    file_idx = episode_info["data/file_index"]
    episode_idx = episode_info["episode_index"]

    parquet_path = (
        data_path / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
    )
    df = pd.read_parquet(parquet_path)

    # Filter to this episode
    ep_data = df[df["episode_index"] == episode_idx].sort_values("frame_index")

    states = np.stack(ep_data["observation.state"].values)
    actions = np.stack(ep_data["action"].values)

    return states, actions


def _get_video_path(data_path: Path, episode_info: dict, camera: str) -> Path:
    """Get the video path for an episode."""
    chunk_key = f"videos/{camera}/chunk_index"
    file_key = f"videos/{camera}/file_index"

    chunk_idx = episode_info[chunk_key]
    file_idx = episode_info[file_key]

    return (
        data_path
        / "videos"
        / camera
        / f"chunk-{chunk_idx:03d}"
        / f"file-{file_idx:03d}.mp4"
    )


def _create_bridge_annotation(
    episode_idx: int,
    states: np.ndarray,
    actions: np.ndarray,
    task_text: str,
    split: str,
    fps_downsample: int = 10,
) -> dict:
    """
    Create annotation JSON.

    LeRobot SO101 format:
    - state: (N, 6) joint positions
    - action: (N-1, 6) relative joint deltas
    - continuous_gripper_state: (N,) last joint (gripper)
    """
    # Downsample to match video fps (30fps → 3fps = every 10th frame)
    states = states[::fps_downsample]
    actions = actions[::fps_downsample]

    # Use native dimensions - no padding
    gripper_state = states[:, -1]  # Last joint is gripper

    # Compute relative actions: action[t] = state[t+1] - state[t]
    relative_actions = np.diff(states, axis=0)  # (N-1, 6)

    return {
        "task": "robot_trajectory_prediction",
        "texts": [task_text],
        "videos": [{"video_path": f"videos/{split}/{episode_idx}/rgb.mp4"}],
        "state": states.tolist(),
        "action": relative_actions.tolist(),
        "continuous_gripper_state": gripper_state.tolist(),
        "episode_id": episode_idx,
    }
