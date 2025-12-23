# src/aloha_cosmos/__init__.py
"""Aloha-Cosmos: Fine-tuning Cosmos world models on ALOHA data."""

__version__ = "0.1.0"

from aloha_cosmos.lerobot_adapter import convert_to_bridge_format
from aloha_cosmos.utils import convert_checkpoint, setup_lerobot_experiment, with_modal

__all__ = [
    "convert_to_bridge_format",
    "convert_checkpoint",
    "setup_lerobot_experiment",
    "with_modal",
]
