#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GRPO implementation.
This script verifies that the GRPO implementation works correctly
with synthetic data, without requiring the actual model.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import GRPO implementation
from src.grpo.grpo import GRPO, GRPOConfig, GRPOPreferenceDataset


def create_synthetic_dataset(size=100, seed=42):
    """
    Create a synthetic preference dataset for testing.
    
    Args:
        size: Number of examples
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    import numpy as np
    np.random.seed(seed)
    
    # Create prompts
    prompts = [
        f"Describe what you see in image {i}" for i in range(size)
    ]
    
    # Create chosen responses (correct answers)
    chosen_responses = [
        f"This image shows a detailed mathematical problem. The answer is {np.random.randint(1, 100)}."
        for _ in range(size)
    ]
    
    # Create rejected responses (incorrect answers)
    rejected_responses = [
        f"I'm not sure what this image shows. Maybe it's {np.random.choice(['A', 'B', 'C', 'D'])}?"
        for _ in range(size)
    ]
    
    # Split into train and eval
    train_size = int(size * 0.8)
    
    train_dataset = GRPOPreferenceDataset(
        prompts=prompts[:train_size],
        chosen_responses=chosen_responses[:train_size],
        rejected_responses=rejected_responses[:train_size],
        shuffle=True,
        seed=seed
    )
    
    eval_dataset = GRPOPreferenceDataset(
        prompts=prompts[train_size:],
        chosen_responses=chosen_responses[train_size:],
        rejected_responses=rejected_responses[train_size:],
        shuffle=False,
        seed=seed
    )
    
    return train_dataset, eval_dataset


def create_mock_tokenizer():
    """
    Create a simple mock tokenizer for testing.
    
    Returns:
        Mock tokenizer object
    """
    class MockTokenizer:
        def encode(self, text):
            # Convert text to simple token IDs (just use character codes)
            return [ord(c) % 32000 for c in text[:512]]
        
        def decode(self, tokens):
            # Convert tokens back to text
            return ''.join([chr(t % 128) for t in tokens if t > 0])
    
    return MockTokenizer()


def test_grpo_initialization():
    """Test GRPO initialization."""
    logger.info("Testing GRPO initialization")
    
    # Create config
    config = GRPOConfig(
        batch_size=4,
        num_epochs=1,
        max_steps=10,
        checkpoint_dir="test_checkpoints"
    )
    
    # Create tokenizer
    tokenizer = create_mock_tokenizer()
    
    # Initialize GRPO
    grpo = GRPO(
        model=None,  # No model for testing
        tokenizer=tokenizer,
        config=config
    )
    
    logger.info("GRPO initialized successfully")
    return grpo


def test_grpo_training(grpo):
    """
    Test GRPO training with synthetic data.
    
    Args:
        grpo: Initialized GRPO instance
    """
    logger.info("Testing GRPO training")
    
    # Create datasets
    train_dataset, eval_dataset = create_synthetic_dataset(size=20)
    
    logger.info(f"Created train dataset with {len(train_dataset)} examples")
    logger.info(f"Created eval dataset with {len(eval_dataset)} examples")
    
    # Train for a few steps
    stats = grpo.train(
        dataset=train_dataset,
        num_epochs=1,
        max_steps=5
    )
    
    logger.info("Training completed")
    logger.info(f"Loss: {stats.get('loss', [])}")
    
    # Evaluate
    metrics = grpo.evaluate(
        test_dataset=eval_dataset,
        num_samples=None  # Use all samples
    )
    
    logger.info("Evaluation completed")
    logger.info(f"Metrics: {metrics}")
    
    return stats, metrics


def test_checkpoint_save_load(grpo):
    """
    Test checkpoint saving and loading.
    
    Args:
        grpo: Trained GRPO instance
    """
    logger.info("Testing checkpoint saving and loading")
    
    # Create checkpoint directory
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    step = 5
    success = grpo.save_checkpoint(step)
    
    if not success:
        logger.error("Failed to save checkpoint")
        return False
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}/checkpoint-{step}")
    
    # Create a new GRPO instance
    new_grpo = test_grpo_initialization()
    
    # Load checkpoint
    success = new_grpo.load_checkpoint(f"{checkpoint_dir}/checkpoint-{step}")
    
    if not success:
        logger.error("Failed to load checkpoint")
        return False
    
    logger.info("Checkpoint loaded successfully")
    
    # Compare stats
    if len(grpo.stats.get('loss', [])) == len(new_grpo.stats.get('loss', [])):
        logger.info("Stats loaded correctly")
    else:
        logger.warning("Stats mismatch")
    
    return True


def clean_up():
    """Clean up test artifacts."""
    logger.info("Cleaning up test artifacts")
    
    # Remove checkpoint directory
    import shutil
    checkpoint_dir = Path("test_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Removed {checkpoint_dir}")


def main():
    """Run all tests."""
    logger.info("Starting GRPO tests")
    
    try:
        # Initialize GRPO
        grpo = test_grpo_initialization()
        
        # Test training
        stats, metrics = test_grpo_training(grpo)
        
        # Test checkpoint save/load
        checkpoint_success = test_checkpoint_save_load(grpo)
        
        logger.info("All tests completed")
        logger.info(f"Training stats: {stats}")
        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(f"Checkpoint test success: {checkpoint_success}")
        
    except Exception as e:
        logger.exception(f"Error during tests: {e}")
        return False
    finally:
        # Clean up
        clean_up()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)