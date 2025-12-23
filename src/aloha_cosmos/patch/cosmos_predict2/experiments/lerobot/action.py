# LeRobot SO101 action-conditioned experiment config
#
# This experiment fine-tunes the action-conditioned model on LeRobot SO101 data.
# Uses registered action-conditioned configs (model/net/conditioner) with mock data,
# then provides explicit LeRobot dataloader. Checkpoint loaded from HuggingFace.

from __future__ import annotations

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]

# LeRobot dataset paths
LEROBOT_PATH = "datasets/lerobot"


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# LeRobot dataset config
lerobot_train_dataset = L(Dataset_3D)(
    train_annotation_path=f"{LEROBOT_PATH}/annotation/train",
    val_annotation_path=f"{LEROBOT_PATH}/annotation/val",
    test_annotation_path=f"{LEROBOT_PATH}/annotation/val",
    video_path=LEROBOT_PATH,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)

lerobot_train_dataloader = L(DataLoader)(
    dataset=lerobot_train_dataset,
    sampler=L(get_sampler)(dataset=lerobot_train_dataset),
    batch_size=2,
    drop_last=True,
)


"""
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=lerobot_so101_2b
"""
lerobot_so101_2b = LazyDict(
    dict(
        defaults=[
            # Use registered configs - don't inherit from DEFAULT_CHECKPOINT.experiment
            # (it references NVIDIA internal datasets not in public release)
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {
                "override /tokenizer": "wan2pt1_tokenizer"
            },  # Use wan2pt1 (available on HuggingFace)
            {
                "override /data_train": "mock"
            },  # Use mock, will be overridden by our dataloader
            {"override /data_val": "mock"},
            {
                "override /ckpt_type": "dcp"
            },  # Use DCP for actual checkpoint saving (not dummy!)
            {"override /callbacks": "wandb"},  # Enable W&B logging
            "_self_",
        ],
        job=dict(
            project="aloha_cosmos",
            group="action_conditioned",
            name="so101_2b",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=500,
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=2000,
            logging_iter=10,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=3,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=500,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=3,
                    save_s3=False,
                ),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,  # Dataset_3D outputs 7D: arm(6) + gripper(1)
                    num_action_per_chunk=12,
                ),
            ),
        ),
        # Use our explicitly defined LeRobot dataloader (not Bridge)
        dataloader_train=lerobot_train_dataloader,
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="lerobot_so101_2b",
    node=lerobot_so101_2b,
)
