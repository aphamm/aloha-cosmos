"""Shared Modal configuration for aloha-cosmos."""

from functools import wraps
from pathlib import Path
from typing import Callable

import modal

CUDA_VERSION = "12.8.1"
PYTHON_VERSION = "3.10"
MODEL_SIZE = "2B"
MOUNT_PATH = Path("/mnt")
COSMOS_LOCAL_DIR = (
    Path(__file__).parent.parent.parent / "external" / "cosmos-predict2.5"
)
PATCH_LOCAL_DIR = Path(__file__).parent / "patch"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    # system dependencies
    .apt_install(
        "ffmpeg",
        "ca-certificates",
        "build-essential",
        "cmake",
        "libsm6",
        "libxext6",
        "curl",
        "pigz",
        "clang",
    )
    .run_commands("python -m pip install --no-cache-dir -U pip uv")
    .add_local_dir(str(COSMOS_LOCAL_DIR), "/cosmos-predict2.5", copy=True)
    .run_commands("cd /cosmos-predict2.5 && uv sync --extra=cu128")
    # environment config
    .workdir("/cosmos-predict2.5")
    .env(
        {
            "HF_HOME": "/mnt/hf_cache",  # cache directory for HF models
            "IMAGINAIRE_OUTPUT_ROOT": "/mnt/checkpoints",  # where training checkpoints and artifacts will be saved
        }
    )
    .uv_pip_install("lerobot==0.4.2")
    # patch files (including .json) - add_local_python_source only includes .py files
    .add_local_dir(str(PATCH_LOCAL_DIR), "/root/aloha_cosmos/patch", copy=True)
    # local source code
    .add_local_python_source("aloha_cosmos")
)


def with_modal(
    app_name: str,
    timeout: int = 1,
    cpu: int | None = None,
    gpu: str | None = None,
) -> Callable:
    """
    Decorator factory for Modal functions with Cosmos configuration.

    Automatically calls setup_modal_env() before the function runs.

    Args:
        timeout: timeout in hours (default: 1 hour)
        cpu: CPU count (for non-GPU functions)
        gpu: GPU type (e.g., "A100-80GB", "A100-40GB", "A100-80GB:2")
    """

    kwargs = {
        "image": image,
        "volumes": {
            MOUNT_PATH: modal.Volume.from_name(
                "aloha-cosmos", create_if_missing=True, version=2
            )
        },
        "secrets": [
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("wandb-secret"),
        ],
        "timeout": timeout * 60 * 60,
    }

    if gpu:
        kwargs["gpu"] = gpu
    if cpu:
        kwargs["cpu"] = cpu

    app = modal.App(app_name)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kw):
            setup_modal_env()
            return fn(*args, **kw)

        return app.function(**kwargs)(wrapper)

    return decorator


def setup_modal_env():
    """Set up environment variables for Cosmos and activate the uv venv."""
    import os
    import site
    import sys

    # Activate the uv virtual environment
    venv_path = Path("/cosmos-predict2.5/.venv")
    site_packages = venv_path / "lib" / "python3.10" / "site-packages"
    if site_packages.exists():
        # Add venv site-packages to the front of sys.path
        site.addsitedir(str(site_packages))
        # Move it to the front so venv packages take precedence
        sys.path.insert(0, str(site_packages))
        os.environ["VIRTUAL_ENV"] = str(venv_path)

    os.environ["COSMOS_CHECKPOINT_DIR"] = "/mnt/checkpoints"

    # Configure wandb for better connectivity in Modal
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")  # 5 minutes
    os.environ.setdefault("WANDB_CONSOLE", "wrap")  # Better error handling

    # Create symlinks as cosmos-predict2.5 repo (expects checkpoints/ and datasets/ in cwd).
    ckpt_link = Path("/cosmos-predict2.5/checkpoints")
    if not (ckpt_link.is_symlink() or ckpt_link.exists()):
        ckpt_link.symlink_to("/mnt/checkpoints")

    data_link = Path("/cosmos-predict2.5/datasets")
    if not (data_link.is_symlink() or data_link.exists()):
        data_link.symlink_to("/mnt/datasets")

    outputs_link = Path("/cosmos-predict2.5/outputs")
    if not (outputs_link.is_symlink() or outputs_link.exists()):
        outputs_link.symlink_to("/mnt/outputs")


def setup_lerobot_experiment():
    """
    Set up LeRobot experiment files at runtime.

    Copies patch files from aloha_cosmos/patch/ to cosmos-predict2.5/.
    The patch directory mirrors the cosmos-predict2.5 structure for easy tracking.
    This runs quickly since it's just file copies, allowing fast iteration.
    """
    import shutil

    patch_root = Path("/root/aloha_cosmos/patch")
    cosmos_root = Path("/cosmos-predict2.5")

    # Copy all patch files, preserving directory structure
    for src in patch_root.rglob("*"):
        if src.is_file():
            # Compute relative path and destination
            rel_path = src.relative_to(patch_root)
            dst = cosmos_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


def convert_checkpoint(
    experiment: str = "cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_action_conditioned",
):
    """
    Convert DCP (Distributed Checkpoint) to consolidated PyTorch format.

    Creates three files in the checkpoint directory:
    - model.pt: Full checkpoint with both regular and EMA weights
    - model_ema_fp32.pt: EMA weights only in float32
    - model_ema_bf16.pt: EMA weights only in bfloat16 (recommended for inference)

    Args:
        experiment: Path relative to IMAGINAIRE_OUTPUT_ROOT (e.g., project/group/name)
    """
    import subprocess

    checkpoints_dir = Path("/mnt/checkpoints") / experiment / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}. "
            "Make sure training has completed."
        )

    # Get iteration from latest_checkpoint.txt if not specified
    latest_file = checkpoints_dir / "latest_checkpoint.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"latest_checkpoint.txt not found in {checkpoints_dir}")
    iteration = latest_file.read_text().strip()

    checkpoint_dir = checkpoints_dir / iteration
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model_dir = checkpoint_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Converting checkpoint: {checkpoint_dir}")

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "./scripts/convert_distcp_to_pt.py",
        str(model_dir),
        str(checkpoint_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Verify output files
    expected_files = ["model.pt", "model_ema_fp32.pt", "model_ema_bf16.pt"]
    for fname in expected_files:
        fpath = checkpoint_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {fpath}")

    print(
        f"Conversion complete! Use checkpoint: {checkpoint_dir / 'model_ema_bf16.pt'}"
    )
    return str(checkpoint_dir / "model_ema_bf16.pt")
