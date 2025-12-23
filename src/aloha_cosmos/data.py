from aloha_cosmos.utils import with_modal


@with_modal(app_name="get-gr1-dataset", cpu=4)
def get_gr1_data():
    """
    Download GR1-100 training dataset from Hugging Face.
    https://huggingface.co/datasets/nvidia/GR1-100

    Dataset folder format should be:

        datasets/benchmark_train/gr1/
        ├── metas/
        │   ├── *.txt
        ├── videos/
        │   ├── *.mp4
        ├── metadata.csv
    """
    import shutil
    import subprocess
    from pathlib import Path

    # Create directory structure
    hf_gr1_dir = Path("/mnt/datasets/benchmark_train/hf_gr1")
    gr1_dir = Path("/mnt/datasets/benchmark_train/gr1")
    videos_dir = gr1_dir / "videos"

    hf_gr1_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    cmd = [
        "/cosmos-predict2.5/.venv/bin/hf",
        "download",
        "nvidia/GR1-100",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(hf_gr1_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Move mp4 files
    gr1_source_dir = hf_gr1_dir / "gr1"
    if gr1_source_dir.exists():
        for mp4_file in gr1_source_dir.glob("*.mp4"):
            dest = videos_dir / mp4_file.name
            shutil.move(str(mp4_file), str(dest))

    # Move metadata.csv
    metadata_src = hf_gr1_dir / "metadata.csv"
    metadata_dst = gr1_dir / "metadata.csv"
    if metadata_src.exists():
        shutil.move(str(metadata_src), str(metadata_dst))

    # Delete hf_gr1_dir
    shutil.rmtree(hf_gr1_dir)

    cmd = [
        "/cosmos-predict2.5/.venv/bin/python",
        "-m",
        "scripts.create_prompts_for_gr1_dataset",
        "--dataset_path",
        str(gr1_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@with_modal(app_name="get-bridge-dataset", cpu=12)
def get_bridge_data():
    """
    Download Bridge training dataset from IRASim.
    https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz

    Each JSON file in annotations/ contains:
    - state: End-effector pose [x, y, z, roll, pitch, yaw]
    - continuous_gripper_state: Gripper width (0=open, 1=closed)
    - action: Gripper displacement (6D pose + 1D open/close binary)
    """
    import shutil
    import subprocess
    from pathlib import Path

    # Create directory structure
    datasets_dir = Path("/mnt/datasets")
    bridge_dir = datasets_dir / "bridge"

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract the dataset
    url = "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz"
    tarball = datasets_dir / "bridge_train_data.tar.gz"

    # Download with retries and resume support for reliability
    print("Downloading bridge dataset...")
    download_cmd = [
        "curl",
        "-L",
        "-C",
        "-",  # Resume partial downloads
        "--retry",
        "5",  # Retry up to 5 times
        "--retry-delay",
        "5",  # Wait 5s between retries
        "--retry-max-time",
        "600",  # Max 10min total retry time
        "-o",
        str(tarball),
        url,
    ]
    subprocess.run(download_cmd, check=True)

    # Extract with pigz for parallel decompression (faster)
    print("Extracting bridge dataset...")
    extract_cmd = f"pigz -dc {tarball} | tar -xf - -C {datasets_dir}"
    subprocess.run(extract_cmd, shell=True, check=True, executable="/bin/bash")

    # Clean up tarball
    tarball.unlink()

    # Move bridge folder from opensource_robotdata to datasets/
    source_dir = datasets_dir / "opensource_robotdata" / "bridge"
    if source_dir.exists():
        if bridge_dir.exists():
            shutil.rmtree(bridge_dir)
        shutil.move(str(source_dir), str(bridge_dir))

    # Clean up opensource_robotdata directory
    opensource_dir = datasets_dir / "opensource_robotdata"
    if opensource_dir.exists():
        shutil.rmtree(opensource_dir)


@with_modal(app_name="get-lerobot-dataset", cpu=4)
def get_lerobot_data(repo_id: str = "aphamm/so101-table-cleanup"):
    """
    Download SO101-Table-Cleanup dataset and convert to Cosmos Bridge format.
    https://huggingface.co/datasets/aphamm/so101-table-cleanup
    """
    import shutil
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from aloha_cosmos import convert_to_bridge_format

    # Create directory structure
    datasets_dir = Path("/mnt/datasets")
    data_dir = datasets_dir / "tmp"
    output_dir = datasets_dir / "lerobot"

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download LeRobot dataset
    print("Step 1: Downloading LeRobot dataset...")
    dataset = LeRobotDataset(repo_id, root=data_dir)

    # Get actual dataset path from the LeRobot object
    dataset_path = Path(dataset.root)
    print(f"Downloaded to: {dataset_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Features: {dataset.meta.features}")

    # Step 2: Convert to Bridge format for Cosmos training
    print("\nStep 2: Converting to Bridge format...")
    convert_to_bridge_format(
        data_path=dataset_path,
        output_path=output_dir,
        camera="observation.images.front",
        train_ratio=0.9,
    )

    # Clean up data directory
    shutil.rmtree(data_dir)

    print(f"Dataset ready for Cosmos training at: {output_dir}")


@with_modal(app_name="validate-data", cpu=2)
def validate_data(data: str = "bridge", max_episodes: int = 1000):
    """
    Validate dataset structure and compute statistics for state, action, and
    continuous gripper state across episodes. Saves statistics to JSON.

    /mnt/datasets/data/
    ├── annotation/{train,val,test}/
    │ └── {episode_id}.json # state, action, texts, video paths
    ├── videos/{train,val,test}/{episode_id}/
    │ ├── rgb.mp4
    │ └── state.npy
    └── latent_videos/ # optional pre-encoded latents

    Args:
        data: Dataset name (e.g., "bridge", "lerobot")
        max_episodes: Max episodes to sample for statistics (0 = all)
    """
    import json
    import subprocess
    from pathlib import Path

    import numpy as np

    data_dir = Path(f"/mnt/datasets/{data}")

    # Check if dataset exists
    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}")
        print(f"Run: modal run src/aloha_cosmos/data.py::get_{data}_data")
        return

    # Validate required structure
    annotations_dir = data_dir / "annotation"
    videos_dir = data_dir / "videos"

    if not annotations_dir.exists():
        raise ValueError(f"Missing: {annotations_dir}")
    if not videos_dir.exists():
        raise ValueError(f"Missing: {videos_dir}")

    # Check train/val splits
    for split in ["train", "val"]:
        ann_dir = annotations_dir / split
        vid_dir = videos_dir / split
        if not ann_dir.exists():
            raise ValueError(f"Missing: {ann_dir}")
        if not vid_dir.exists():
            raise ValueError(f"Missing: {vid_dir}")

    print(f"Dataset: {data_dir}")

    # Collect all data across episodes for statistics
    all_states = []
    all_actions = []
    all_gripper_states = []

    total_sampled = 0
    for split in ["train", "val"]:
        ann_dir = annotations_dir / split
        annotation_files = sorted(ann_dir.glob("*.json"))
        print(f"{split}: {len(annotation_files)} episodes")

        for annotation_file in annotation_files:
            if max_episodes > 0 and total_sampled >= max_episodes:
                break
            total_sampled += 1
            with open(annotation_file, "r") as f:
                annotation = json.load(f)

            if "state" in annotation:
                states = annotation["state"]
                if states and isinstance(states[0], list):
                    all_states.extend(states)
                elif states:
                    all_states.append(states)

            if "action" in annotation:
                actions = annotation["action"]
                if actions and isinstance(actions[0], list):
                    all_actions.extend(actions)
                elif actions:
                    all_actions.append(actions)

            if "continuous_gripper_state" in annotation:
                gripper = annotation["continuous_gripper_state"]
                if isinstance(gripper, list):
                    all_gripper_states.extend(gripper)

    # Compute statistics
    statistics = {}

    if all_states:
        states_arr = np.array(all_states)
        state_stats = {
            "min": states_arr.min(axis=0).tolist(),
            "max": states_arr.max(axis=0).tolist(),
            "mean": states_arr.mean(axis=0).tolist(),
            "std": states_arr.std(axis=0).tolist(),
        }
        statistics["state"] = state_stats
        print("State Statistics (per dimension):")
        print(f"{'Dim':<5} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
        for i in range(states_arr.shape[1]):
            print(
                f"{i:<5} {state_stats['min'][i]:>12.6f} {state_stats['max'][i]:>12.6f} "
                f"{state_stats['mean'][i]:>12.6f} {state_stats['std'][i]:>12.6f}"
            )

    if all_actions:
        actions_arr = np.array(all_actions)
        action_stats = {
            "min": actions_arr.min(axis=0).tolist(),
            "max": actions_arr.max(axis=0).tolist(),
            "mean": actions_arr.mean(axis=0).tolist(),
            "std": actions_arr.std(axis=0).tolist(),
        }
        statistics["action"] = action_stats
        print("Action Statistics (per dimension):")
        print(f"{'Dim':<5} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
        for i in range(actions_arr.shape[1]):
            print(
                f"{i:<5} {action_stats['min'][i]:>12.6f} {action_stats['max'][i]:>12.6f} "
                f"{action_stats['mean'][i]:>12.6f} {action_stats['std'][i]:>12.6f}"
            )

    if all_gripper_states:
        gripper_arr = np.array(all_gripper_states)
        gripper_stats = {
            "min": float(gripper_arr.min()),
            "max": float(gripper_arr.max()),
            "mean": float(gripper_arr.mean()),
            "std": float(gripper_arr.std()),
        }
        statistics["continuous_gripper_state"] = gripper_stats
        print("Continuous Gripper State Statistics:")
        print(
            f"Min: {gripper_stats['min']:.6f}, Max: {gripper_stats['max']:.6f}, "
            f"Mean: {gripper_stats['mean']:.6f}, Std: {gripper_stats['std']:.6f}"
        )

    # Save statistics to JSON
    stats_file = data_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    # Analyze first episode video
    episode_id = 0
    train_vid = videos_dir / "train"
    episode_dir = train_vid / str(episode_id)

    print(f"\nSample Episode {episode_dir}:")

    # Analyze rgb.mp4
    rgb_file = episode_dir / "rgb.mp4"
    if rgb_file.exists():
        print("Video: rgb.mp4")
        print(f"Size: {rgb_file.stat().st_size / 1024:.1f} KB")

        # Get video metadata using ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(rgb_file),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                for stream in video_info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        print(f"Codec: {stream.get('codec_name', 'unknown')}")
                        print(
                            f"Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}"
                        )
                        print(f"Frame rate: {stream.get('r_frame_rate', 'unknown')}")
                        duration = stream.get("duration", 0)
                        if duration:
                            print(f"Duration: {float(duration):.2f}s")
                        print(f"Frames: {stream.get('nb_frames', 'unknown')}")
                        print(f"Pixel format: {stream.get('pix_fmt', 'unknown')}")

                fmt = video_info.get("format", {})
                bitrate = fmt.get("bit_rate", 0)
                if bitrate:
                    print(f"Bitrate: {int(bitrate) / 1000:.0f} kbps")
        except Exception as e:
            print(f"Could not get video metadata: {e}")

    # Analyze latent if exists
    latent_file = data_dir / "latent_videos" / "train" / str(episode_id) / "0.pt"
    if latent_file.exists():
        import torch

        print(f"\nLatent: {latent_file}")
        print(f"Size: {latent_file.stat().st_size / 1024:.1f} KB")
        latent = torch.load(latent_file, weights_only=True)
        print(f"Shape: {latent.shape}")
        print(f"Dtype: {latent.dtype}")

        # Summary statistics per channel (across T, H, W)
        print("Latent Statistics (per channel):")
        print(f"{'Ch':<5} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        for c in range(latent.shape[1]):
            ch = latent[:, c, :, :].float()
            print(
                f"{c:<5} {ch.min():>10.4f} {ch.max():>10.4f} {ch.mean():>10.4f} {ch.std():>10.4f}"
            )
