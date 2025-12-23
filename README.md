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

- [Modal account](https://modal.com/) for serverless GPU

- [HuggingFace account](https://huggingface.co) for model checkpoints and datasets

- [Wandb account](https://wandb.ai/site/) for experiment tracking

- Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1)

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
modal run src/aloha_cosmos/data.py::get_bridge_data
```

### Run LoRa Finetuning

```sh
modal run --detach src/aloha_cosmos/train.py::train_lora
```

### Run Video2World Inference

```bash
# base model
modal run --detach src/aloha_cosmos/generate.py::video2world --id 0 \
    --prompt "Use the right hand to pick up rubik's cube from top level of the wooden shelf to bottom level of the wooden shelf."

# lora model
modal run --detach src/aloha_cosmos/generate.py::video2world --id 0 --use-lora --seed 42 \
    --prompt "Use the right hand to pick up rubik's cube from top level of the wooden shelf to bottom level of the wooden shelf." \
    --dit-path "checkpoints/posttraining/video2world/2b_groot_gr1_480/checkpoints/model/iter_000002000.pt"
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
