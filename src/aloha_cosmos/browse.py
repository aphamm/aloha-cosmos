"""Inspect dataset structure and contents on Modal."""

from aloha_cosmos import setup_modal_env, with_modal


@with_modal(app_name="inspect-bridge", cpu=2)
def inspect_bridge():
    """
    Analyze the file hierarchy of /mnt/datasets/bridge and inspect the first video
    with all its components (video file + annotation JSON).

    Dataset structure:
        /mnt/datasets/bridge/
        ├── annotation/
        │   ├── test/
        │   │   ├── 0.json, 1.json, ...
        │   ├── train/
        │   └── val/
        ├── videos/
        │   ├── test/
        │   │   ├── 0/
        │   │   │   ├── rgb.mp4
        │   │   │   └── state.npy
        │   │   ├── 1/
        │   │   │   ├── rgb.mp4
        │   │   │   └── state.npy
        │   ├── train/
        │   └── val/
        └── latent_videos/
            ├── test/
            ├── train/
            └── val/
    """
    import json
    import subprocess
    from pathlib import Path

    import numpy as np
    import torch

    setup_modal_env()

    bridge_dir = Path("/mnt/datasets/bridge")

    # Check if bridge dataset exists
    if not bridge_dir.exists():
        print("Bridge dataset not found at /mnt/datasets/bridge")
        print("Run: modal run src/aloha_cosmos/data.py::get_bridge_data")
        return

    # Show overall structure with nested directories
    print(bridge_dir)

    for top_dir in sorted(bridge_dir.iterdir()):
        if top_dir.is_dir():
            print(f"  \__ {top_dir.name}/")

            # Show subdirectories (train/test/val)
            for sub_dir in sorted(top_dir.iterdir()):
                if sub_dir.is_dir():
                    # Count episode directories or files
                    items = list(sub_dir.iterdir())
                    episode_count = len([d for d in items if d.is_dir()])
                    file_count = len([f for f in items if f.is_file()])

                    if episode_count > 0:
                        print(f"      \__ {sub_dir.name}/ ({episode_count} episodes)")
                        # Show first episode directory
                        episode_dirs = sorted(
                            [d for d in items if d.is_dir()],
                            key=lambda x: int(x.name) if x.name.isdigit() else x.name,
                        )
                        first_ep = episode_dirs[0]
                        ep_files = sorted(first_ep.iterdir())
                        print(f"          \__ {first_ep.name}/")
                        for f in ep_files[:3]:
                            print(f"              \__ {f.name}")
                    else:
                        print(f"      \__ {sub_dir.name}/ ({file_count} files)")
                        sorted_items = sorted(items)[:3]
                        for f in sorted_items:
                            print(f"          \__ {f.name}")

    # Analyze videos and annotations directories (with train/test/val splits)
    videos_dir = bridge_dir / "videos"
    annotations_dir = bridge_dir / "annotation"

    train_videos_dir = videos_dir / "train"
    train_annotations_dir = annotations_dir / "train"

    episode_id = 0

    if train_videos_dir.exists():
        episode_dir = train_videos_dir / str(episode_id)
        if episode_dir.exists():
            print(f"\nAnalyze episode {episode_id}...\n")
            print(f"   Path: {episode_dir}")

            # Analyze rgb.mp4
            rgb_file = episode_dir / "rgb.mp4"
            if rgb_file.exists():
                print("\n   Video: rgb.mp4")
                print(f"   Size: {rgb_file.stat().st_size / 1024:.1f} KB")

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
                                print(
                                    f"   Codec: {stream.get('codec_name', 'unknown')}"
                                )
                                print(
                                    f"   Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}"
                                )
                                print(
                                    f"   Frame rate: {stream.get('r_frame_rate', 'unknown')}"
                                )
                                duration = stream.get("duration", 0)
                                if duration:
                                    print(f"   Duration: {float(duration):.2f}s")
                                print(
                                    f"   Frames: {stream.get('nb_frames', 'unknown')}"
                                )
                                print(
                                    f"   Pixel format: {stream.get('pix_fmt', 'unknown')}"
                                )

                        fmt = video_info.get("format", {})
                        bitrate = fmt.get("bit_rate", 0)
                        if bitrate:
                            print(f"   Bitrate: {int(bitrate) / 1000:.0f} kbps")
                except Exception as e:
                    print(f"   Could not get video metadata: {e}")

            # Analyze state.npy
            state_file = episode_dir / "state.npy"
            if state_file.exists():
                print("\n   State: state.npy")
                print(f"   Size: {state_file.stat().st_size / 1024:.1f} KB")

                data = np.load(state_file)
                print(f"   Shape: {data.shape}")
                print(f"   Dtype: {data.dtype}")

            # Find corresponding annotation
            annotation_file = train_annotations_dir / f"{episode_id}.json"

            if annotation_file.exists():
                print(f"\n   Annotation: {annotation_file.name}")
                print(f"   Size: {annotation_file.stat().st_size / 1024:.1f} KB")

                with open(annotation_file, "r") as f:
                    annotation = json.load(f)

                # Print annotation with dimensionality for large arrays
                for key, value in annotation.items():
                    if key in (
                        "action",
                        "state",
                        "continuous_gripper_state",
                    ) and isinstance(value, list):
                        # Show dimensionality instead of full data
                        if value and isinstance(value[0], list):
                            print(f"   {key}: ({len(value)} x {len(value[0])})")
                        else:
                            print(f"   {key}: ({len(value)})")
                    else:
                        print(f"   {key}: {value}")

            else:
                print(f"\n   No annotation found for episode {episode_id}")

            # Analyze latent_video
            latent_videos_dir = bridge_dir / "latent_videos" / "train"
            latent_ep_dir = latent_videos_dir / str(episode_id)
            if latent_ep_dir.exists():
                latent_file = latent_ep_dir / "0.pt"
                if latent_file.exists():
                    print(f"\n   Latent: {latent_file}")
                    print(f"   Size: {latent_file.stat().st_size / 1024:.1f} KB")

                    data = torch.load(
                        latent_file, map_location="cpu", weights_only=True
                    )
                    if isinstance(data, torch.Tensor):
                        print(f"   Shape: {data.shape}")
                        print(f"   Dtype: {data.dtype}")
