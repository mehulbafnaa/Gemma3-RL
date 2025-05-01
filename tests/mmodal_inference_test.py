
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- JAX and Gemma Imports ---
try:
    import jax
    import jax.numpy as jnp
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available JAX devices: {jax.devices()}")
    jax.config.update('jax_default_matmul_precision', 'bfloat16')
    logger.info("JAX configured with 'bfloat16' matmul precision.")
except ImportError:
    logger.error("Error importing JAX. Please ensure JAX is installed.")
    sys.exit(1)

try:
    from gemma import gm
    logger.info("Successfully imported 'gemma' library.")
except ImportError:
    logger.error("Error importing the 'gemma' library. Please ensure it's installed.")
    sys.exit(1)
# --- End Imports ---


def ensure_consistent_dtypes(params: Any, target_dtype=jnp.bfloat16) -> Any:
    """
    Recursively ensures all floating-point arrays in parameters have the target dtype.
    Args:
        params: Parameter tree.
        target_dtype: Target JAX data type for floating-point arrays.
    Returns:
        Parameter tree with floating-point arrays converted.
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


def preprocess_image(image_path: str, target_size: int = 896) -> Optional[jnp.ndarray]:
    """
    CORRECTED image preprocessing for Gemma 3 multimodal input.

    Returns a 4D JAX array [1, H, W, C] (uint8).
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.is_file():
        logger.error(f"Image file not found: {image_path}")
        return None

    try:
        logger.info(f"Preprocessing image: {image_path}")
        img = Image.open(image_path)
        if img.mode != "RGB": img = img.convert("RGB")
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)

        # Ensure 3D [H, W, C] before adding batch dim
        if img_array.ndim == 2:  # Grayscale
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.ndim == 4: # Attempt to fix 4D
            if img_array.shape[-1] == 4: img_array = img_array[:, :, :3]  # RGBA -> RGB
            elif img_array.shape[0] == 1: img_array = img_array.squeeze(axis=0) # Squeeze batch
            else: img_array = img_array[0] # Take first frame/slice

        if not (img_array.ndim == 3 and img_array.shape == (target_size, target_size, 3)):
            logger.error(f"Image array shape error after processing: {img_array.shape}")
            return None

        # Convert to JAX array and add BATCH dimension -> [1, H, W, C]
        img_jnp = jnp.array(img_array, dtype=jnp.uint8)
        img_jnp = jnp.expand_dims(img_jnp, axis=0) # CORRECT 4D Output

        logger.info(f"Image preprocessed successfully to shape {img_jnp.shape} [1, H, W, C].") # Log correct shape
        return img_jnp

    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file: {image_path}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error preprocessing image {image_path}: {e}", exc_info=True)
        return None

# --- CORRECTED TYPE HINT FOR 'model' ---
def get_model_context_length(model: Any) -> int:
    """ Attempts to dynamically determine the model's context length. """
    default_context_length = 8192 # Fallback
    possible_attr_names = ['context_length', 'max_position_embeddings', 'seq_length']
    # Check model.config first
    config_source = getattr(model, 'config', None)
    if config_source:
        for attr_name in possible_attr_names:
            context_length = getattr(config_source, attr_name, None)
            if isinstance(context_length, int) and context_length > 0:
                 return context_length
    # Check model attributes directly
    for attr_name in possible_attr_names:
        context_length = getattr(model, attr_name, None)
        if isinstance(context_length, int) and context_length > 0:
            return context_length

    logger.warning(f"Could not determine context length. Using default: {default_context_length}")
    return default_context_length


def run_multimodal_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str, # This prompt WILL be used now
    image_path: str,
    model_size: str = "4b",
    max_new_tokens: int = 256,
    seed: int = 42
) -> Tuple[Optional[str], Optional[float]]:
    """
    Runs CORRECTED multimodal inference using Gemma 3.
    """
    logger.info("Starting multimodal inference run...")
    if '<img>' not in prompt:
         logger.warning("Prompt does not contain '<img>' tag. Image may be ignored.")

    start_total_time = time.time()
    tokens_per_second = None
    model, tokenizer, params, sampler = None, None, None, None

    try:
        # === Load Assets ===
        logger.info("Loading model parameters...")
        loaded_params = gm.ckpts.load_params(path=checkpoint_path, text_only=False)
        params = ensure_consistent_dtypes(loaded_params); del loaded_params

        logger.info("Loading tokenizer...")
        tokenizer = gm.text.Gemma3Tokenizer(path=str(tokenizer_path))

        logger.info(f"Creating model instance (Gemma3_{model_size})...")
        if model_size.lower() == "4b": model = gm.nn.Gemma3_4B()
        elif model_size.lower() == "9b": model = gm.nn.Gemma3_9B()
        # Add elif for other supported sizes (e.g., "27b") if needed
        # elif model_size.lower() == "27b": model = gm.nn.Gemma3_27B()
        else: raise ValueError(f"Unsupported model size: {model_size}")

        logger.info("Creating ChatSampler...")
        sampler = gm.text.ChatSampler(model=model, params=params, tokenizer=tokenizer, multi_turn=False)

        # === Preprocess Image (Outputs 4D Tensor) ===
        logger.info("Preprocessing image...")
        image_processed = preprocess_image(image_path)
        if image_processed is None: raise ValueError("Image preprocessing failed.")

        # === Prepare Generation ===
        key = jax.random.PRNGKey(seed)
        if max_new_tokens == -1: # Dynamic calculation
            model_context_len = get_model_context_length(model)
            input_ids = tokenizer.encode(prompt)
            estimated_image_tokens = 256 # Heuristic
            available_len = model_context_len - len(input_ids) - estimated_image_tokens
            actual_max_new_tokens = max(1, available_len - 20) # Buffer
        else:
            actual_max_new_tokens = max(1, max_new_tokens) # Ensure positive

        # === Run Generation (Corrected Call) ===
        logger.info(f"Generating response (max_new_tokens={actual_max_new_tokens})...")
        logger.info(f"Using image shape: {image_processed.shape}") # Expect [1, H, W, C]
        gen_start_time = time.time()

        # --- CORRECTED CALL ---
        output_text = sampler.chat(
            prompt=prompt,           # Use the input prompt with <img>
            images=image_processed,  # Pass the 4D tensor DIRECTLY
            rng=key,
            max_new_tokens=actual_max_new_tokens
        )
        # --- END CORRECTED CALL ---

        gen_elapsed = time.time() - gen_start_time
        logger.info(f"Generation finished ({gen_elapsed:.2f}s).")

        if not output_text: output_text = "[Model produced no text]"
        if gen_elapsed > 0: tokens_per_second = actual_max_new_tokens / gen_elapsed

    except (FileNotFoundError, ValueError, UnidentifiedImageError) as e:
        logger.error(f"Setup or Input Error: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"Runtime error during inference: {e}", exc_info=True)
        if "out of memory" in str(e).lower() or "resourceexhaustederror" in str(e).lower():
            logger.error("OOM error detected during generation.")
        return None, None
    finally:
        # Attempt to clean up large JAX/Flax objects from memory
        del params, model, sampler, image_processed
        logger.debug("Cleaned up model assets.")

    total_elapsed = time.time() - start_total_time
    logger.info(f"Total inference run time: {total_elapsed:.2f} seconds.")
    return output_text, tokens_per_second


def main():
    """ Main function: derives paths, parses args, runs inference. """
    parser = argparse.ArgumentParser(
        description="Run Gemma 3 multimodal inference using relative paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt including '<img>'.")
    parser.add_argument("--model_size", type=str, default="4b", choices=["4b", "9b"], help="Gemma 3 model size.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens (-1 for dynamic).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # === Path Derivation ===
    try:
        script_dir = Path(__file__).parent.resolve()
        # ASSUMPTION: Script is in a subdirectory (like 'tests') of the project root.
        # If script is AT the project root, change script_dir.parent -> script_dir
        project_root = script_dir.parent
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Assumed project root: {project_root}")

        # Define relative paths (adjust if your structure differs)
        checkpoint_dir_name = f"gemma3-{args.model_size}" # e.g., gemma3-4b
        checkpoint_path = project_root / "pre-trained" / checkpoint_dir_name
        tokenizer_path = project_root / "pre-trained" / "tokenizer.model"
        images_dir = project_root / "data" / "images"

        logger.info(f"Derived checkpoint path: {checkpoint_path}")
        logger.info(f"Derived tokenizer path: {tokenizer_path}")
        logger.info(f"Derived images directory: {images_dir}")

        if not images_dir.is_dir(): raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Find first image file (sorted for consistency)
        image_files = sorted(list(images_dir.glob('*.jpg'))) + \
                      sorted(list(images_dir.glob('*.png'))) + \
                      sorted(list(images_dir.glob('*.jpeg')))
        if not image_files: raise FileNotFoundError(f"No images found in {images_dir}")

        image_path = image_files[0]
        logger.info(f"Using first found image: {image_path.name}")

        # Validate essential paths before proceeding
        if not checkpoint_path.is_dir(): raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        if not tokenizer_path.is_file(): raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    except FileNotFoundError as e:
        logger.error(f"Path Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error setting up paths: {e}", exc_info=True)
        sys.exit(1)
    # === End Path Derivation ===

    # Run inference
    result_text, tok_sec = run_multimodal_inference(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt=args.prompt, # Pass the prompt from command line
        image_path=str(image_path),
        model_size=args.model_size,
        max_new_tokens=args.max_tokens,
        seed=args.seed
    )

    # Print results
    print("\n" + "="*70)
    if result_text:
        print("Multimodal Output:")
        print("-"*70)
        print(result_text)
        print("-"*70)
        if tok_sec is not None: print(f"Performance: ~{tok_sec:.2f} tokens/second")
    else:
        print("Multimodal Inference Failed.")
        print("Please check logs for errors.")
    print("="*70)

if __name__ == "__main__":
    main()

jax.clear_caches()
