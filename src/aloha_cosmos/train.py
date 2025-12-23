from aloha_cosmos import setup_modal_env, with_modal


@with_modal(app_name="train-lora", gpu="A100-80GB:2", timeout=16)
def train_lora(
    experiment: str = "predict2_video2world_training_2b_groot_gr1_480",
):
    """
    Run LoRA post-training on the Cosmos-Predict2 model.
    """
    import subprocess

    setup_modal_env()

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=12341",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/configs/base/config.py",
        "--",
        f"experiment={experiment}",
        # LoRA architecture
        "model.config.train_architecture=lora",
        "model.config.lora_rank=16",
        "model.config.lora_alpha=16",
        # Disable prompt refiner/guardrail during training
        "model.config.pipe_config.prompt_refiner_config.enabled=False",
        "model.config.pipe_config.guardrail_config.enabled=False",
        "dataloader_train.batch_size=1",
        # Training settings
        "trainer.max_iter=2000",
        "trainer.logging_iter=10",
        "checkpoint.save_iter=400",
        # Optimizer (higher LR for LoRA)
        "optimizer.lr=0.0009765625",  # 2^(-10)
        # Scheduler warmup
        "scheduler.warm_up_steps=[0]",
        "scheduler.cycle_lengths=[2000]",
        "scheduler.f_max=[0.6]",
        "scheduler.f_min=[0.0]",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
