# src/aloha_cosmos/__init__.py
"""Aloha-Cosmos: Fine-tuning Cosmos world models on ALOHA data."""

__version__ = "0.1.0"

from aloha_cosmos.utils import setup_modal_env, with_modal

__all__ = [
    "setup_modal_env",
    "with_modal",
]
