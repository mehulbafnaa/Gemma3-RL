#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 Model Instance Factory Module

This module provides utility functions for creating and managing
Gemma 3 model instances for both inference and training with RL algorithms.
It offers a centralized way to initialize, load, and configure model components.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import jax
import jax.numpy as jnp

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def ensure_consistent_dtypes(params: Any, target_dtype=jnp.bfloat16) -> Any:
    """
    Ensure all floating-point arrays have consistent datatypes.
    
    Args:
        params: Parameter tree to convert
        target_dtype: Target JAX data type for floating-point arrays
        
    Returns:
        Parameter tree with floating-point arrays converted to target dtype
    """
    if isinstance(params, dict):
        return {k: ensure_consistent_dtypes(v, target_dtype) for k, v in params.items()}
    elif isinstance(params, (list, tuple)):
        return type(params)(ensure_consistent_dtypes(v, target_dtype) for v in params)
    elif hasattr(params, 'dtype') and jnp.issubdtype(params.dtype, jnp.floating):
        if params.dtype != target_dtype:
            return jnp.asarray(params, dtype=target_dtype)
        return params
    return params

def load_gemma_tokenizer(tokenizer_path: str) -> Any:
    """
    Load the Gemma 3 tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer model file
        
    Returns:
        Loaded tokenizer or None if loading fails
    """
    try:
        from gemma import text as gm_text
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = gm_text.Gemma3Tokenizer(path=str(tokenizer_path))
        return tokenizer
    except ImportError:
        logger.error("Failed to import gemma.text. Make sure gemma library is installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None

def load_gemma_parameters(checkpoint_path: str) -> Optional[Dict]:
    """
    Load Gemma 3 model parameters with proper dtype handling.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        
    Returns:
        Model parameters dictionary or None if loading fails
    """
    try:
        from gemma import checkpoints as gm_ckpts
        logger.info(f"Loading parameters from: {checkpoint_path}")
        
        # Load parameters
        loaded_params = gm_ckpts.load_params(path=checkpoint_path, text_only=False)
        
        # Ensure consistent dtypes
        params = ensure_consistent_dtypes(loaded_params)
        
        # Clear memory
        del loaded_params
        
        return params
    except ImportError:
        logger.error("Failed to import gemma.checkpoints. Make sure gemma library is installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        return None

def create_gemma_model(model_size: str = "4b") -> Optional[Any]:
    """
    Create a Gemma 3 model instance.
    
    Args:
        model_size: Model size variant ("4b", "12b", or "27b")
        
    Returns:
        Gemma model instance or None if creation fails
    """
    try:
        from gemma import nn as gm_nn
        logger.info(f"Creating Gemma3_{model_size} model instance")
        
        # Create model instance based on size
        if model_size.lower() == "4b":
            model = gm_nn.Gemma3_4B()
        elif model_size.lower() == "12b":
            model = gm_nn.Gemma3_12B()
        elif model_size.lower() == "27b":
            model = gm_nn.Gemma3_27B()
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
            
        return model
    except ImportError:
        logger.error("Failed to import gemma.nn. Make sure gemma library is installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return None

# --- Main Model Instance Functions ---
def create_model_instance(
    checkpoint_path: str,
    tokenizer_path: str,
    model_size: str = "4b",
    seed: int = 42,
    multi_turn: bool = False
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Create a complete Gemma 3 model instance with all components.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        tokenizer_path: Path to tokenizer model file
        model_size: Model size variant ("4b", "12b", or "27b")
        seed: Random seed for initialization
        multi_turn: Whether to enable multi-turn conversation mode
        
    Returns:
        Tuple of (model, params, tokenizer, sampler) or Nones on failure
    """
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(seed)
    
    try:
        # Set JAX precision for matrix operations
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
        
        # Load components
        tokenizer = load_gemma_tokenizer(tokenizer_path)
        params = load_gemma_parameters(checkpoint_path)
        model = create_gemma_model(model_size)
        
        # Verify components were loaded successfully
        if tokenizer is None or params is None or model is None:
            logger.error("Failed to load one or more model components")
            return None, None, None, None
            
        # Create sampler for text generation
        try:
            from gemma import text as gm_text
            logger.info(f"Creating ChatSampler (multi_turn={multi_turn})")
            sampler = gm_text.ChatSampler(
                model=model,
                params=params,
                tokenizer=tokenizer,
                multi_turn=multi_turn
            )
        except ImportError:
            logger.error("Failed to import gemma.text for sampler. Make sure gemma library is installed.")
            sampler = None
        except Exception as e:
            logger.error(f"Failed to create sampler: {e}")
            sampler = None
            
        return model, params, tokenizer, sampler
    
    except Exception as e:
        logger.exception(f"Error creating model instance: {e}")
        return None, None, None, None

def preprocess_image_for_model(
    image: Union[str, np.ndarray],
    target_size: int = 896
) -> Optional[jnp.ndarray]:
    """
    Preprocess an image for Gemma 3 model input.
    
    Args:
        image: Path to image file or numpy array
        target_size: Target size for the square image
        
    Returns:
        JAX array with shape [1, H, W, C] and appropriate dtype
    """
    try:
        # If image is a string, it's a path to an image file
        if isinstance(image, str):
            try:
                from PIL import Image
                img = Image.open(image)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.uint8)
            except Exception as e:
                logger.error(f"Failed to load/process image from path: {e}")
                return None
        # If image is a numpy array, use it directly
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            raise TypeError(f"Expected image to be string or numpy array, got {type(image)}")
            
        # Handle different input shapes
        if img_array.ndim == 2:  # Grayscale
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.ndim == 3:
            if img_array.shape[-1] == 1:  # Single channel
                img_array = np.concatenate((img_array,) * 3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[..., :3]
            elif img_array.shape[-1] != 3:
                raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")
        else:
            raise ValueError(f"Unexpected image dimensions: {img_array.ndim}")
            
        # Resize if needed
        if img_array.shape[0] != target_size or img_array.shape[1] != target_size:
            from PIL import Image
            img = Image.fromarray(img_array)
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
            
        # Convert to JAX array and add batch dimension
        img_jnp = jnp.array(img_array, dtype=jnp.uint8)
        img_jnp = jnp.expand_dims(img_jnp, axis=0)
        
        logger.info(f"Image preprocessed to shape {img_jnp.shape} with dtype {img_jnp.dtype}")
        return img_jnp
    
    except Exception as e:
        logger.exception(f"Error preprocessing image: {e}")
        return None

def generate_with_image(
    sampler: Any,
    prompt: str,
    image: Optional[Union[str, np.ndarray]] = None,
    max_new_tokens: int = 256,
    seed: int = 42
) -> Optional[str]:
    """
    Generate text from the model using a prompt and optionally an image.
    
    Args:
        sampler: Gemma sampler instance
        prompt: Text prompt
        image: Optional image path or array
        max_new_tokens: Maximum number of new tokens to generate
        seed: Random seed for generation
        
    Returns:
        Generated text or None on failure
    """
    try:
        # Set seed
        key = jax.random.PRNGKey(seed)
        
        # Process image if provided
        image_processed = None
        if image is not None:
            image_processed = preprocess_image_for_model(image)
            if image_processed is None:
                logger.warning("Failed to process image, continuing without it")
                
        # Generate text
        logger.info(f"Generating with max_new_tokens={max_new_tokens}")
        gen_start_time = time.time()
        
        output_text = sampler.chat(
            prompt=prompt,
            images=image_processed,
            max_new_tokens=max_new_tokens,
            rng=key
        )
        
        gen_elapsed = time.time() - gen_start_time
        logger.info(f"Generation completed in {gen_elapsed:.2f}s")
        
        # Calculate performance
        if output_text and gen_elapsed > 0:
            tokens_per_second = max_new_tokens / gen_elapsed
            logger.info(f"Performance: ~{tokens_per_second:.2f} tokens/second")
            
        return output_text
    
    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        return None
    finally:
        # Clean up resources
        jax.clear_caches()

def get_model_context_length(model: Any) -> int:
    """
    Attempt to dynamically determine the model's context length.
    
    Args:
        model: Gemma model instance
        
    Returns:
        Context length as integer
    """
    default_context_length = 128000
    possible_attr_names = ['context_length', 'max_position_embeddings', 'seq_length']
    
    # Check model.config first
    if hasattr(model, 'config'):
        for attr_name in possible_attr_names:
            context_length = getattr(model.config, attr_name, None)
            if isinstance(context_length, int) and context_length > 0:
                logger.info(f"Found context length in model.config.{attr_name}: {context_length}")
                return context_length
    
    # Check model attributes directly
    for attr_name in possible_attr_names:
        context_length = getattr(model, attr_name, None)
        if isinstance(context_length, int) and context_length > 0:
            logger.info(f"Found context length in model.{attr_name}: {context_length}")
            return context_length
    
    logger.warning(f"Could not determine context length. Using default: {default_context_length}")
    return default_context_length

# --- Demo Usage ---
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Paths to model components
    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = str(project_root / "pre-trained/gemma3-4b")
    tokenizer_path = str(project_root / "pre-trained/tokenizer.model")
    
    # Sample prompt and image path
    prompt = "Describe what you see in this image."
    image_path = str(project_root / "data/images/sample.jpg")
    
    print(f"Project root: {project_root}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    
    # Create model instance
    model, params, tokenizer, sampler = create_model_instance(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_size="4b",
        seed=42
    )
    
    if all([model, params, tokenizer, sampler]):
        print("Model instance created successfully")
        
        # Generate text
        output = generate_with_image(
            sampler=sampler,
            prompt=prompt,
            image=image_path if Path(image_path).exists() else None,
            max_new_tokens=256
        )
        
        if output:
            print("\nGenerated Output:")
            print("-" * 50)
            print(output)
            print("-" * 50)
        else:
            print("Failed to generate output")
    else:
        print("Failed to create model instance")