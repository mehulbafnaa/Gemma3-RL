# Gemma3-RL: Fine-tuning Gemma3 with Reinforcement Learning

This repository implements Group Relative Policy Optimization (GRPO) for fine-tuning the Gemma 3 multimodal model on mathematical reasoning tasks. The implementation focuses on improving the model's ability to solve mathematical problems that involve both text and visual information using the MathVista dataset.

## Overview

This project adapts GRPO (a type of RLHF algorithm) specifically for multimodal inputs to improve Gemma 3's capabilities in visual-mathematical reasoning. The key components include:

- A custom reward function for mathematical reasoning that evaluates format adherence, answer correctness, and reasoning quality
- A complete training pipeline for fine-tuning Gemma 3 models on the MathVista dataset
- Support for multimodal (text + image) inputs and mathematical problem-solving
- Integration with JAX and TPU acceleration for efficient training

## Repository Structure

- **configs/** - Configuration files for model and training settings
  - `gemma3_4B_config.yaml` - Gemma 3 model configuration
  - `grpo_train_config.yaml` - GRPO training configuration
- **data/** - Dataset storage directory
  - `images/` - Directory for MathVista image files
- **pre-trained/** - Scripts and storage for pre-trained model files
  - `download_model.sh` - Script to download Gemma 3 models
  - `unzip_model.sh` - Script to extract model files
  - `tokenizer.model` - Gemma 3 tokenizer model file
- **src/** - Source code directory
  - `data/` - Dataset loading and preprocessing
    - `dataset.py` - MathVista dataset implementation
    - `preprocessing.py` - Utilities for preprocessing inputs/outputs
    - `prompt_template.py` - Templates for formatting prompts
  - `grpo/` - GRPO algorithm implementation
    - `grpo.py` - Core GRPO trainer implementation
  - `model/` - Model interfaces
    - `model_instance.py` - Interface to the Gemma 3 model
  - `reward/` - Reward functions
    - `math_reward.py` - Mathematical reasoning reward functions
  - `utils/` - Utility functions
    - `checkpoint.py` - Checkpoint management
    - `logging.py` - Logging utilities
    - `tpu_utils.py` - TPU support utilities
- **tests/** - Test files
  - `dataset_test.py` - Tests for dataset loading
  - `test_grpo.py` - Tests for GRPO implementation
  - `gemma3_multimodal_test.sh` - Multimodal model test script
- **run_grpo.py** - Main script for running GRPO training
- **train_mathvista_grpo.py** - Script for training on MathVista dataset
- **GRPO_IMPLEMENTATION_NOTES.md** - Implementation notes and status
- **requirements.txt** - Python dependencies

## Requirements

- Python 3.12
- JAX (with TPU support)
- Flax
- Transformers
- Orbax (for checkpointing)
- TensorFlow (for dataset processing)
- Pillow (for image processing)
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Gemma3-RL.git
   cd Gemma3-RL
   ```

2. Set up a virtual environment and install dependencies using `uv`:
   ```bash
   # Install uv if not already installed
   pip install uv

   # Create and activate a virtual environment with Python 3.12
   uv venv -p python3.12
   source .venv/bin/activate  # On Linux/macOS
   # or
   # .venv\Scripts\activate  # On Windows

   # Install dependencies
   uv pip install -r requirements.txt
   ```

3. Download the pre-trained Gemma 3 model:
   ```bash
   cd pre-trained
   ./download_model.sh
   ./unzip_model.sh
   cd ..
   ```

4. Prepare the MathVista dataset:
   - Download the MathVista dataset
   - Place image files in `data/images/`
   - Update dataset paths in configuration files if needed

## Usage

### Training

To train the model with GRPO on the MathVista dataset:

```bash
python train_mathvista_grpo.py --config configs/grpo_train_config.yaml
```

You can modify the training parameters in `configs/grpo_train_config.yaml` to adjust:
- Learning rate, batch size, and training steps
- GRPO-specific parameters (beta, margin)
- Model size and configuration
- Dataset settings

### Evaluation

To evaluate a trained model:

```bash
python run_grpo.py --config configs/grpo_train_config.yaml --mode eval --checkpoint_path /path/to/checkpoint
```

### Model Architecture

This implementation uses the Gemma 3 model, which consists of:
- Language model component with 34 transformer layers (4B model)
- Vision model component with 27 transformer layers
- Visual-language integration via cross-attention mechanisms

## GRPO Implementation

The GRPO algorithm is implemented in `src/grpo/grpo.py`. Key features include:

1. **Reward Function**: The mathematical reasoning reward function evaluates:
   - Format adherence (proper use of reasoning and answer tags)
   - Answer correctness (exact match, numerical match, or token overlap)
   - Reasoning quality (length, step indicators, mathematical notation)

2. **Training Loop**:
   - Processes batches of examples from the dataset
   - Generates multiple responses for each prompt
   - Computes rewards for each response
   - Updates model parameters using the GRPO objective

3. **Preference Learning**:
   - Uses pairs of chosen (preferred) and rejected (non-preferred) responses
   - Optimizes policy to increase probability of chosen responses
   - Maintains KL divergence penalty to prevent policy collapse

## Current Development State

This branch (`gemma3-grpo-dev`) contains the ongoing implementation of GRPO for Gemma3 fine-tuning. Key features that have been implemented:

- Core GRPO algorithm implementation in `src/grpo/grpo.py`
- Reward functions for mathematical reasoning in `src/reward/math_reward.py`
- Dataset handling for MathVista's multimodal inputs
- Training script integration with JAX/TPU

Known issues and ongoing work:
- Orbax checkpoint compatibility issues
- Model integration with the latest Gemma 3 API
- Reward function refinement for better mathematical reasoning
- Testing with real model checkpoints and MathVista data

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{gemma3-rl,
  title={Gemma3-RL: Fine-tuning Gemma3 with Reinforcement Learning from Preferences for Multimodal Mathematical Reasoning},
  author={Your Name},
  year={2025}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgements

- Based on the Gemma 3 model by Google DeepMind
- Uses the MathVista dataset for evaluation
- Implements GRPO algorithm for multimodal preference learning