from aloha_cosmos.utils import setup_modal_env, with_modal


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

    setup_modal_env()

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

    Dataset folder format should be:

        datasets/bridge/
        ├── annotations/
        │   ├── *.json
        ├── videos/
            ├── *.mp4

    Each JSON file in annotations/ contains:
    - state: End-effector pose [x, y, z, roll, pitch, yaw]
    - continuous_gripper_state: Gripper width (0=open, 1=closed)
    - action: Gripper displacement (6D pose + 1D open/close binary)

    We use this information as conditioning input for video generation
    """
    import shutil
    import subprocess
    from pathlib import Path

    setup_modal_env()

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
