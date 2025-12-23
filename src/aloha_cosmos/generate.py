"""Inference for action-conditioned video generation."""

from pathlib import Path

from aloha_cosmos import convert_checkpoint, setup_lerobot_experiment, with_modal


@with_modal(app_name="generate-action-bridge", gpu="A100-80GB")
def generate_action_bridge(
    experiment: str = "cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_action_conditioned",
):
    """
    Run action-conditioned video generation inference with Video2World model finetuned on Bridge dataset.
    """
    import subprocess

    checkpoints_dir = Path("/mnt/checkpoints") / experiment / "checkpoints"
    iteration = (checkpoints_dir / "latest_checkpoint.txt").read_text().strip()
    checkpoint_dir = checkpoints_dir / iteration
    checkpoint_path = str(checkpoint_dir / "model_ema_bf16.pt")

    if not Path(checkpoint_path).exists():
        checkpoint_path = convert_checkpoint(experiment)

    print(f"Using checkpoint: {checkpoint_path}")

    output_dir = f"/mnt/outputs/action_conditioned/{experiment.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "examples/action_conditioned.py",
        "-i",
        "assets/action_conditioned/basic/inference_params.json",
        "-o",
        output_dir,
        "--save_root",
        output_dir,
        "--checkpoint-path",
        checkpoint_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\nGeneration complete! Videos saved to: {output_dir}")


@with_modal(app_name="generate-action-lerobot", gpu="A100-80GB")
def generate_action_lerobot(
    experiment: str = "aloha_cosmos/action_conditioned/so101_2b",
):
    """
    Run action-conditioned video generation inference with Video2World model finetuned on LeRobot SO101 dataset.
    """
    import subprocess

    checkpoints_dir = Path("/mnt/checkpoints") / experiment / "checkpoints"
    iteration = (checkpoints_dir / "latest_checkpoint.txt").read_text().strip()
    checkpoint_dir = checkpoints_dir / iteration
    checkpoint_path = str(checkpoint_dir / "model_ema_bf16.pt")

    if not Path(checkpoint_path).exists():
        checkpoint_path = convert_checkpoint(experiment)

    print(f"Using checkpoint: {checkpoint_path}")

    output_dir = f"/mnt/outputs/action_conditioned/{experiment.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    setup_lerobot_experiment()

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "examples/action_conditioned.py",
        "-i",
        "assets/action_conditioned/lerobot/inference_params.json",
        "-o",
        output_dir,
        "--save_root",
        output_dir,
        "--checkpoint-path",
        checkpoint_path,
        "--experiment",
        "lerobot_so101_2b",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\nGeneration complete! Videos saved to: {output_dir}")
