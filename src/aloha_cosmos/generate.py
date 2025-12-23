from pathlib import Path

from aloha_cosmos.utils import MODEL_SIZE, setup_modal_env, with_modal


@with_modal(app_name="aloha-cosmos-generate", gpu="A100-80GB")
def video2world(
    id: int = 0,
    prompt: str = "",
    negative_prompt: str = "",
    fps: int = 16,
    resolution: int = 720,
    seed: int = -1,
    use_lora: bool = False,
    dit_path: str = "",
):
    """
    Generate a video from an input image using Cosmos Predict2 Video2World.

    Can use either the base model or a LoRA finetuned model.

    Args:
        id: ID for input/output files (expects image_{id}.png as input on volume)
        prompt: Text description for video generation
        negative_prompt: Things to avoid in generation
        fps: Frames per second for output video
        resolution: Video resolution (480 or 720)
        seed: Random seed (-1 for random)
        use_lora: If True, use LoRA finetuned model instead of base model
        dit_path: Path to LoRA checkpoint (relative to repo root)
    """
    import subprocess

    setup_modal_env()

    # Read input image from volume
    input_image_path = Path(f"/mnt/outputs/image_{id}.png")
    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}.")

    if use_lora:
        output_path = Path(f"/mnt/outputs/lora_{id}.mp4")

        # Use LoRA finetuned model
        lora_path = Path("/cosmos-predict2.5") / dit_path
        if not lora_path.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint not found: {lora_path}. "
                "Make sure training has completed and checkpoint exists."
            )

        print(f"Generating video with LoRA model: {dit_path}")
        cmd = [
            "/cosmos-predict2.5/.venv/bin/python",
            "-m",
            "examples.video2world_lora",
            "--model_size",
            MODEL_SIZE,
            "--dit_path",
            str(dit_path),
            "--input_path",
            str(input_image_path),
            "--prompt",
            prompt,
            "--save_path",
            str(output_path),
            "--use_lora",
            "--lora_rank",
            "16",
            "--lora_alpha",
            "16",
            "--lora_target_modules",
            "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            "--offload_guardrail",
            "--offload_prompt_refiner",
            "--fps",
            str(fps),
            "--resolution",
            str(resolution),
        ]
    else:
        output_path = Path(f"/mnt/outputs/base_{id}.mp4")

        # Use base model
        print("Generating video with base model")
        cmd = [
            "/cosmos-predict2.5/.venv/bin/python",
            "-m",
            "examples.video2world",
            "--model_size",
            MODEL_SIZE,
            "--input_path",
            str(input_image_path),
            "--prompt",
            prompt,
            "--save_path",
            str(output_path),
            "--fps",
            str(fps),
            "--resolution",
            str(resolution),
        ]

    if negative_prompt:
        cmd.extend(["--negative_prompt", negative_prompt])
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])

    print(f"   Input: {input_image_path}")
    print(f"   Prompt: {prompt}")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Video saved to volume: {output_path}")
