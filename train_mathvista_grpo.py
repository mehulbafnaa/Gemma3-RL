#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathVista GRPO training script for Gemma 3-4B model.

This script trains a Gemma 3-4B model on the MathVista dataset using
Group Relative Policy Optimization (GRPO) to improve mathematical reasoning.
"""

import logging
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from PIL import Image

from src.grpo.grpo import GRPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image, target_size=896):
    """
    Prepare an image for the model
    
    Args:
        image: PIL image or path to image
        target_size: Size to resize to
        
    Returns:
        JAX array of the processed image
    """
    try:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            # If it's already a PIL image, just make sure it's RGB
            img = image.convert("RGB")
            
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        img_jnp = jnp.array(img_array, dtype=jnp.uint8)
        img_jnp = jnp.expand_dims(img_jnp, axis=0)
        return img_jnp
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

def prepare_mathvista_dataset(dataset, max_examples=None):
    """
    Prepare MathVista dataset for GRPO training
    
    Args:
        dataset: HuggingFace dataset
        max_examples: Maximum number of examples to use (for debugging)
        
    Returns:
        List of formatted examples
    """
    formatted_dataset = []
    
    # Limit examples for debugging if needed
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    logger.info(f"Preparing {len(dataset)} examples from MathVista dataset")
    
    for i, item in enumerate(dataset):
        try:
            # Get question and answer
            question = item["question"].strip()
            answer = item["answer"]
            
            # Process image
            image = item.get("decoded_image")
            if image:
                image_data = preprocess_image(image)
            else:
                logger.warning(f"Example {i} has no image, skipping")
                continue
                
            # Create formatted example
            example = {
                "question": question,
                "answer": answer,
                "image": image_data,
                "pid": item.get("pid", f"example_{i}"),
                "answer_type": item.get("answer_type", "unknown"),
                "precision": item.get("precision", None),
            }
            
            formatted_dataset.append(example)
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} examples")
                
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
    
    logger.info(f"Successfully prepared {len(formatted_dataset)} examples")
    return formatted_dataset

def main():
    """Main function to run GRPO training on MathVista dataset"""
    try:
        # Determine project paths - FIXED to use the correct paths
        current_dir = Path(__file__).parent.absolute()
        
        # Use absolute paths to avoid any confusion
        model_path = current_dir / "pre-trained" / "gemma3-4b"
        tokenizer_path = current_dir / "pre-trained" / "tokenizer.model"
        output_dir = current_dir / "outputs" / "grpo-mathvista"
        
        # Print path info
        logger.info("\nPATH INFO:")
        logger.info(f"  Current directory: {current_dir}")
        logger.info(f"  Checkpoint: {model_path}")
        logger.info(f"  Tokenizer: {tokenizer_path}")
        logger.info(f"  Output dir: {output_dir}")
        
        # Verify paths exist
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")
            
        if not tokenizer_path.exists():
            logger.error(f"Tokenizer path does not exist: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
        
        # Configure JAX
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        
        # Load MathVista dataset
        logger.info("Loading MathVista dataset...")
        dataset = load_dataset("AI4Math/MathVista", split="testmini")
        
        # Prepare dataset for training
        # Use a small subset for initial testing
        formatted_dataset = prepare_mathvista_dataset(dataset, max_examples=10)
        
        # Initialize GRPO trainer
        logger.info("Initializing GRPO trainer...")
        trainer = GRPOTrainer(
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path),
            output_dir=str(output_dir),
            num_generations=2,  # Start with smaller number for testing
            max_seq_length=1024,
            max_prompt_length=256,
            learning_rate=5e-6,
            kl_coef=0.1,
            format_weight=1.0,
            answer_weight=2.0,
            reasoning_weight=1.0,
            logging_steps=1,
            save_steps=5
        )
        
        # Run training for a few steps initially
        logger.info("Starting GRPO training...")
        trainer.train(
            dataset=formatted_dataset,
            max_steps=10,  # Small number of steps for initial testing
            batch_size=1
        )
        
        # Test the trained model on a few examples
        logger.info("Testing trained model...")
        test_indices = [0, 1, 2]  # Test on first three examples
        
        for idx in test_indices:
            test_item = formatted_dataset[idx]
            question = test_item["question"]
            answer = test_item.get("answer", "")
            image = test_item.get("image", None)
            
            logger.info(f"\nTesting example {idx} (pid: {test_item['pid']}):")
            response, metrics = trainer.generate_math_example(question, answer, image)
            
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"Ground Truth: {answer}")
            logger.info(f"Response: {response[:200]}...")
            
            if metrics:
                logger.info(f"Metrics: {metrics}")
        
        logger.info("GRPO training and testing completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())