
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 4B Model for MathVista - No Arguments Required

A simplified implementation for running the Gemma 3 4B model on MathVista dataset
with no command-line arguments required.
"""

import logging
import sys
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from gemma import gm
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure JAX
jax.config.update('jax_default_matmul_precision', 'bfloat16')

class GemmaTokens:
    """Gemma 3 special tokens"""
    START_OF_TURN = "<start_of_turn>"
    END_OF_TURN = "<end_of_turn>"
    START_OF_IMAGE = "<start_of_image>"
    USER = "user"
    MODEL = "model"

def format_prompt(text_prompt, include_image=True):
    """Format a text prompt with Gemma tokens"""
    tokens = GemmaTokens()
    prompt = f"{tokens.START_OF_TURN}{tokens.USER}\n"
    prompt += f"Solve this mathematical problem step-by-step and give me all the reasoning traces. Format your final answer as <answer>YOUR ANSWER</answer>\n"
    prompt += f"{text_prompt}\n"
    
    if include_image:
        prompt += f"{tokens.START_OF_IMAGE}\n"
        
    prompt += f"{tokens.END_OF_TURN}\n{tokens.START_OF_TURN}{tokens.MODEL}"
    return prompt

def preprocess_image(image_path, target_size=896):
    """Prepare an image for the model"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        img_jnp = jnp.array(img_array, dtype=jnp.uint8)
        img_jnp = jnp.expand_dims(img_jnp, axis=0)
        return img_jnp
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

def ensure_float16(params):
    """Convert floating point arrays to bfloat16"""
    if isinstance(params, dict):
        return {k: ensure_float16(v) for k, v in params.items()}
    elif isinstance(params, (list, tuple)):
        return type(params)(ensure_float16(v) for v in params)
    elif hasattr(params, 'dtype') and jnp.issubdtype(params.dtype, jnp.floating):
        if params.dtype != jnp.bfloat16:
            return jnp.asarray(params, dtype=jnp.bfloat16)
    return params

def get_mathvista_example():
    """
    Get a random example from the MathVista dataset.
    
    Returns:
        Tuple of (question, image_path, example_id) or (None, None, None) if no examples found
    """
    try:

                
        # Attempt to load MathVista testmini split
        dataset = load_dataset("AI4Math/MathVista", split="test")
        
        if not dataset or len(dataset) == 0:
            logger.error("MathVista dataset is empty or failed to load")
            return None, None, None
        
        # Select random example
        example = random.choice(dataset)
        
        # Get question and example ID
        question = example.get("question", "").strip()
        example_id = example.get("pid", "unknown_id")
        
        # Get image path - extract just the filename from the metadata
        image_filename = Path(example.get("image", "")).name
        
        # Determine project paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        images_dir = project_root / "data" / "images"
        
        # Create images directory if it doesn't exist
        images_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = images_dir / image_filename
        
        if not image_path.exists():
            logger.error(f"Image not found at {image_path}")
            return question, None, example_id
        
        logger.info(f"Found example {example_id} with image {image_path}")
        return question, str(image_path), example_id
        
    except Exception as e:
        logger.error(f"Error getting MathVista example: {e}")
        return None, None, None

def main():
    """Main function to run the Gemma 3 4B model on a MathVista example"""
    try:
        # Determine project paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        
        checkpoint_path = project_root / "pre-trained" / "gemma3-4b"
        tokenizer_path = project_root / "pre-trained" / "tokenizer.model"
        
        # Print path info
        print("\nPATH INFO:")
        print(f"  Project root: {project_root}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Tokenizer: {tokenizer_path}")
        
        # Get MathVista example
        question, image_path, example_id = get_mathvista_example()
        
        if not question:
            logger.error("Failed to get a valid MathVista example")
            return
        
        # Format the prompt
        formatted_prompt = format_prompt(
            text_prompt=question,
            include_image=(image_path is not None)
        )
        
        # Print the example info
        print(f"\nEXAMPLE ID: {example_id}")
        if image_path:
            print(f"IMAGE: {image_path}")
        
        # Print the question and formatted prompt
        print("\nQUESTION:")
        print("-" * 50)
        print(question)
        print("-" * 50)
        
        print("\nFORMATTED PROMPT:")
        print("-" * 50)
        print(formatted_prompt)
        print("-" * 50)
        
        # Load the model components
        logger.info("Loading model components...")
        
        tokenizer = gm.text.Gemma3Tokenizer(path=str(tokenizer_path))
        model = gm.nn.Gemma3_4B()
        params = gm.ckpts.load_params(path=str(checkpoint_path), text_only=False)
        params = ensure_float16(params)
        
        sampler = gm.text.ChatSampler(
            model=model,
            params=params,
            tokenizer=tokenizer,
            multi_turn=False
        )
        
        # Process the image if available
        image_processed = None
        if image_path:
            logger.info(f"Processing image: {image_path}")
            image_processed = preprocess_image(image_path)
            if image_processed is None:
                logger.warning("Image processing failed, running in text-only mode")
        
        # Generate output
        logger.info("Generating response...")
        start_time = time.time()
        
        seed = 42
        key = jax.random.PRNGKey(seed)
        output_text = sampler.chat(
            prompt=formatted_prompt,
            images=image_processed,
            rng=key
        )
        
        elapsed_time = time.time() - start_time
        
        # Print the result
        print("\n" + "="*70)
        print("MODEL OUTPUT:")
        print("-"*70)
        print(output_text)
        print("-"*70)
        print(f"Generation time: {elapsed_time:.2f} seconds")
        print("="*70)
        
        # Clean up
        jax.clear_caches()
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()