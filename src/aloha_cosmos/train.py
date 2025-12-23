from aloha_cosmos import setup_lerobot_experiment, with_modal


@with_modal(app_name="train-lora-gr1", gpu="A100-80GB:2", timeout=16)
def train_lora_gr1():
    """
    https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_gr00t.md
    """
    import subprocess

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=12341",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/configs/video2world/config.py",
        "--",
        "experiment=predict2_video2world_training_2b_groot_gr1_480",
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


@with_modal(app_name="train-action-bridge", gpu="A100-80GB:2", timeout=16)
def train_action_bridge():
    """
    https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md
    """
    import subprocess

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=12341",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        "experiment=ac_reason_embeddings_rectified_flow_2b_256_320",
        "~dataloader_train.dataloaders",
        # Training settings
        "trainer.max_iter=2000",
        "checkpoint.save_iter=500",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@with_modal(app_name="train-action-lerobot", gpu="A100-80GB:2", timeout=16)
def train_action_lerobot():
    """
    Train action-conditioned video generation on SO101 LeRobot dataset.

    Requires: modal run src/aloha_cosmos/data.py::get_lerobot_data
    """
    import subprocess

    setup_lerobot_experiment()

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=12341",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        "experiment=lerobot_so101_2b",
        # Remove inherited dataloaders to prevent Bridge data mixing
        "~dataloader_train.dataloaders",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
