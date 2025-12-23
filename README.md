# Aloha Cosmos 🌊

Run NVIDIA's Cosmos Predict2 world foundation models on Modal's serverless GPUs. Supports Text2Image, Video2World, and action-conditioned models.

## Getting Started

Instructions on running this project locally.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) an extremely fast Python package and project manager

```sh
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- [Modal](https://modal.com/) for serverless GPU

- [HuggingFace](https://huggingface.co) for model checkpoints and datasets

- [Wandb](https://wandb.ai/site/) for experiment tracking

- Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1)

- Accept the [Cosmos-Predict2.5 License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)

### Setting Up

```sh
# clone repo with submodules
git clone --recurse-submodules git@github.com:aphamm/aloha-cosmos.git
# install dependencies
uv sync
# install package locally
uv pip install -e .
```

```sh
modal setup
# create hugginface token: https://huggingface.co/settings/tokens
modal secret create huggingface-secret HF_TOKEN=your_api_key_here
# create wandb token: https://wandb.ai/authorize
modal secret create wandb-secret WANDB_API_KEY=your_api_key_here
```

## Usage

### Prepare Datasets

Download and preprocess dataset to a persistent Modal Volume.

```sh
# download and preprocess GR1-100 dataset
modal run src/aloha_cosmos/data.py::get_gr1_data

# download Bridge training dataset
#   state: (N, 8) end-effector poses [x, y, z, roll, pitch, yaw, gripper]
#   action: (N-1, 7) relative EE actions [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
#   continuous_gripper_state: scalar gripper per frame
modal run src/aloha_cosmos/data.py::get_bridge_data

# download Lerobot training dataset
modal run src/aloha_cosmos/data.py::get_lerobot_data

# ensure proper metadata
modal run src/aloha_cosmos/data.py::validate_data --data=lerobot
```

### Finetune Action-Conditioned Models

```sh
# Video2World Post-training for DreamGen Bench
modal run --detach src/aloha_cosmos/train.py::train_lora_gr1
# Video2World Post-training for Action-conditioned Video Prediction
modal run --detach src/aloha_cosmos/train.py::train_action_bridge
modal run --detach src/aloha_cosmos/train.py::train_action_lerobot
```

### Run Video2World Inference

````bash
modal run src/aloha_cosmos/generate.py::generate_action_bridge

modal run src/aloha_cosmos/generate.py::generate_action_lerobot
```

## Roadmap

- [x] Enable GR1-100 finetuning
- [ ] Enable Aloha Unleashed action conditioned finetuning
- [ ] Enable epic-kitchens-100 finetuning

## Acknowledgments

- [cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [aloha-unleashed](https://aloha-unleashed.github.io)
- [epic-kitchens-100](https://epic-kitchens.github.io/2025)

## Citation

If you find this repository useful, please consider citing it as:

```bibtex
@misc{pham2025alohacosmos,
      title={Aloha Cosmos: Fine-tuning World Models for Egocentric Robot Policy Evaluation},
      author={Austin Pham and Hod Lipson and Yue Wang},
      journal={arXiv preprint},
      year={2025},
}
```
````
