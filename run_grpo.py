#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for running GRPO (Group Relative Policy Optimization) fine-tuning
on the Gemma3 multimodal model.

This script:
1. Loads model, tokenizer, and dataset configuration
2. Prepares the training and evaluation datasets
3. Initializes the GRPO algorithm
4. Runs the training process
5. Evaluates the model
6. Saves the results
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import jax
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"grpo_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.model.model_instance import create_model_instance
    from src.data.dataset import MathVistaDataset
    from src.grpo.grpo import GRPO, GRPOConfig, GRPOPreferenceDataset
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    raise


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set JAX seed
    key = jax.random.PRNGKey(seed)
    
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate essential configuration keys
        required_keys = ["model", "training", "data"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def create_preference_dataset(
    dataset: MathVistaDataset,
    split_size: float = 0.8,
    seed: int = 42,
) -> Tuple[GRPOPreferenceDataset, GRPOPreferenceDataset]:
    """
    Create preference datasets for training and evaluation.
    
    In a real implementation, this would use a real preference dataset.
    For this placeholder implementation, we create a synthetic preference dataset
    from the MathVista dataset.
    
    Args:
        dataset: MathVista dataset
        split_size: Proportion of the dataset to use for training
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Creating preference datasets with split size {split_size}")
    
    examples = dataset.get_examples()
    
    # Shuffle examples
    rng = np.random.RandomState(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    
    # Split into train and eval
    split_idx = int(len(examples) * split_size)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    # Extract data for train set
    train_prompts = []
    train_chosen = []
    train_rejected = []
    train_images = []
    
    for idx in train_indices:
        example = examples[idx]
        train_prompts.append(example["gemma_prompt"])
        
        # For the placeholder implementation, we use the ground truth answer as the chosen response
        # and a dummy rejected response
        train_chosen.append(f"The answer is {example['answer']}")
        train_rejected.append("I'm not sure about the answer.")
        
        # Get image path
        image_path = example["image_path"]
        if Path(image_path).exists():
            # In a real implementation, this would load the image into a numpy array
            train_images.append(None)  # Placeholder for image
        else:
            train_images.append(None)
    
    # Extract data for eval set
    eval_prompts = []
    eval_chosen = []
    eval_rejected = []
    eval_images = []
    
    for idx in eval_indices:
        example = examples[idx]
        eval_prompts.append(example["gemma_prompt"])
        
        # Same approach as train set
        eval_chosen.append(f"The answer is {example['answer']}")
        eval_rejected.append("I'm not sure about the answer.")
        
        # Get image path
        image_path = example["image_path"]
        if Path(image_path).exists():
            # In a real implementation, this would load the image into a numpy array
            eval_images.append(None)  # Placeholder for image
        else:
            eval_images.append(None)
    
    # Create datasets
    train_dataset = GRPOPreferenceDataset(
        prompts=train_prompts,
        chosen_responses=train_chosen,
        rejected_responses=train_rejected,
        images=train_images,
        shuffle=True,
        seed=seed,
    )
    
    eval_dataset = GRPOPreferenceDataset(
        prompts=eval_prompts,
        chosen_responses=eval_chosen,
        rejected_responses=eval_rejected,
        images=eval_images,
        shuffle=False,
        seed=seed,
    )
    
    logger.info(f"Created train dataset with {len(train_dataset)} examples")
    logger.info(f"Created eval dataset with {len(eval_dataset)} examples")
    
    return train_dataset, eval_dataset


def main(args: argparse.Namespace) -> None:
    """
    Main function for running GRPO fine-tuning.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config["training"].get("seed", 42))
    
    # Print JAX device info
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Determine paths
    model_path = config["model"].get("path")
    tokenizer_path = config["model"].get("tokenizer_path")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = None, None  # Placeholder for the actual model loading
    
    try:
        model, model_params, tokenizer, sampler = create_model_instance(
            checkpoint_path=model_path,
            tokenizer_path=tokenizer_path,
            model_size=config["model"].get("size", "4b"),
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Prepare dataset
    logger.info("Preparing dataset")
    
    data_dir = config["data"].get("data_dir", "data")
    images_dir = config["data"].get("images_dir", "data/images")
    
    dataset = None  # Placeholder for the actual dataset
    
    try:
        dataset = MathVistaDataset(
            data_dir=data_dir,
            images_dir=images_dir,
            split=config["data"].get("split", "testmini"),
            batch_size=config["training"].get("batch_size", 4),
        )
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise
    
    # Create preference datasets
    train_dataset, eval_dataset = create_preference_dataset(
        dataset=dataset,
        split_size=config["data"].get("train_split", 0.8),
        seed=config["training"].get("seed", 42),
    )
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        # Learning parameters
        learning_rate=config["training"].get("learning_rate", 1e-5),
        weight_decay=config["training"].get("weight_decay", 0.01),
        
        # Training parameters
        batch_size=config["training"].get("batch_size", 4),
        accumulation_steps=config["training"].get("accumulation_steps", 1),
        num_epochs=config["training"].get("num_epochs", 3),
        max_steps=config["training"].get("max_steps", 1000),
        
        # GRPO specific
        reference_free=config["training"].get("reference_free", False),
        beta=config["training"].get("beta", 0.1),
        margin=config["training"].get("margin", 0.0),
        
        # System
        seed=config["training"].get("seed", 42),
        
        # Saving and logging
        log_interval=config["training"].get("log_interval", 10),
        save_interval=config["training"].get("save_interval", 100),
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    # Initialize GRPO
    logger.info("Initializing GRPO")
    
    grpo = GRPO(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
    )
    
    # Train the model
    logger.info("Starting training")
    
    start_time = time.time()
    
    try:
        stats = grpo.train(
            dataset=train_dataset,
            num_epochs=config["training"].get("num_epochs", 3),
            max_steps=config["training"].get("max_steps", 1000),
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # Evaluate the model
    logger.info("Evaluating model")
    
    try:
        metrics = grpo.evaluate(
            test_dataset=eval_dataset,
            num_samples=config["training"].get("eval_samples", None),
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    
    # Save results
    logger.info("Saving results")
    
    try:
        results = {
            "metrics": metrics,
            "training_time": training_time,
            "config": config,
        }
        
        with open(output_dir / "results.yaml", "w") as f:
            yaml.dump(results, f)
        
        logger.info(f"Results saved to {output_dir / 'results.yaml'}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise
    
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning")
    parser.add_argument("--config", type=str, default="configs/grpo_train_config.yaml", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.exception(f"Error running GRPO fine-tuning: {e}")
        raise