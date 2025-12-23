"""Shared Modal configuration for aloha-cosmos."""

from pathlib import Path
from typing import Callable

import modal

CUDA_VERSION = "12.8.1"
PYTHON_VERSION = "3.10"
MODEL_SIZE = "2B"
MOUNT_PATH = Path("/mnt")

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    # system dependencies
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "ca-certificates",
        "build-essential",
        "cmake",
        "libsm6",
        "libxext6",
        "curl",
        "pigz",
    )
    .run_commands("git lfs install")
    .run_commands("python -m pip install --no-cache-dir -U pip uv")
    # external repos
    .run_commands(
        "git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git /cosmos-predict2.5"
    )
    .run_commands("cd /cosmos-predict2.5 && git lfs pull")
    .run_commands("cd /cosmos-predict2.5 && uv sync --extra=cu128")
    # environment config
    .workdir("/cosmos-predict2.5")
    .env(
        {
            "HF_HOME": "/mnt/hf_cache",  # cache directory for HF models
            "IMAGINAIRE_OUTPUT_ROOT": "/mnt/checkpoints",  # where training checkpoints and artifacts will be saved
        }
    )
    # pinned dependencies
    .add_local_file("requirements.modal.txt", "/requirements.modal.txt", copy=True)
    .run_commands("uv pip install --no-cache -r /requirements.modal.txt --system")
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
    return app.function(**kwargs)


def setup_modal_env():
    """Set up environment variables for Cosmos."""
    import os

    os.environ["COSMOS_CHECKPOINT_DIR"] = "/mnt/checkpoints"

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
