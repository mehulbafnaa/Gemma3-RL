#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 Multimodal Inference Test Script

This script provides a clean interface for testing Gemma 3's 
multimodal capabilities, handling image+text inputs with proper
data type handling and optimized inference.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path to enable imports
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))

# --- Import from project modules ---
try:
    # Import from our preprocessing module
    from src.data.preprocessing import resize_image_for_gemma, extract_answer_from_response
    # Import from prompt templates (assuming we created it)
    from src.data.prompt_template import create_mathvista_prompt, GemmaTokens
    logger.info("Successfully imported project modules")
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.warning("Continuing with local definitions, but consider creating the required modules")
    
    # Define fallback classes/functions if imports fail
    class GemmaTokens:
        """String representations of Gemma 3 special tokens"""
        BOS = "<bos>"
        EOS = "<eos>"
        START_OF_TURN = "<start_of_turn>"
        END_OF_TURN = "<end_of_turn>"
        START_OF_IMAGE = "<start_of_image>"
        END_OF_IMAGE = "<end_of_image>"
        USER = "user"
        MODEL = "model"
    
    def create_mathvista_prompt(question, instruction="Solve this problem step-by-step", include_image=True):
        """Fallback prompt creation function"""
        tokens = GemmaTokens()
        prompt = f"{tokens.START_OF_TURN}{tokens.USER}\n{question}\n"
        if include_image:
            prompt += f"{tokens.START_OF_IMAGE}\n"
        prompt += f"{tokens.END_OF_TURN}\n{tokens.START_OF_TURN}{tokens.MODEL}"
        return prompt

# --- JAX and Gemma Import Section ---
try:
    import jax
    import jax.numpy as jnp
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
    # Set precision for matrix operations
    jax.config.update('jax_default_matmul_precision', 'bfloat16')
    logger.info("JAX configured with 'bfloat16' matmul precision")
except ImportError as e:
    logger.error(f"Error importing JAX libraries: {e}")
    sys.exit(1)

try:
    from gemma import gm
    logger.info("Successfully imported 'gemma' library")
except ImportError as e:
    logger.error(f"Error importing 'gemma' library: {e}")
    sys.exit(1)

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

def preprocess_image_for_inference(image_path: str, target_size: int = 896) -> Optional[jnp.ndarray]:
    """
    Prepare image for Gemma 3, using resize_image_for_gemma if available.
    
    Args:
        image_path: Path to image file
        target_size: Target size for the square image
        
    Returns:
        JAX array with shape [1, H, W, C] with uint8 dtype
    """
    try:
        # Use imported function if available, otherwise use fallback
        if 'resize_image_for_gemma' in globals():
            # Get raw image array
            img_array = resize_image_for_gemma(image_path, target_size)
            if img_array is None:
                raise ValueError("Image processing failed")
                
            # Convert to JAX array and add batch dimension
            img_jnp = jnp.array(img_array, dtype=jnp.uint8)
            img_jnp = jnp.expand_dims(img_jnp, axis=0)
            
            logger.info(f"Image preprocessed to shape {img_jnp.shape} with dtype {img_jnp.dtype}")
            return img_jnp
        
        # Fallback implementation (copy of previous code)
        logger.info(f"Using fallback image processing for: {image_path}")
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        
        # Handle different input shapes
        if img_array.ndim == 2:  # Grayscale
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.ndim == 4:  # Extra dimensions
            if img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            elif img_array.shape[0] == 1:
                img_array = img_array.squeeze(axis=0)
            else:
                img_array = img_array[0]
        
        # Verify correct shape
        if not (img_array.ndim == 3 and img_array.shape == (target_size, target_size, 3)):
            raise ValueError(f"Image shape error after processing: {img_array.shape}")
            
        # Convert to JAX array and add batch dimension
        img_jnp = jnp.array(img_array, dtype=jnp.uint8)
        img_jnp = jnp.expand_dims(img_jnp, axis=0)
        
        logger.info(f"Image preprocessed to shape {img_jnp.shape} with dtype {img_jnp.dtype}")
        return img_jnp
    
    except Exception as e:
        logger.exception(f"Error preprocessing image: {e}")
        return None

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

# --- Main Inference Function ---
def run_multimodal_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str,
    image_path: str,
    model_size: str = "4b",
    max_new_tokens: int = 256,
    seed: int = 42,
    multi_turn: bool = False
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run multimodal inference with Gemma 3 using both text and image inputs.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        tokenizer_path: Path to tokenizer model file
        prompt: Text prompt for the model
        image_path: Path to image file
        model_size: Size variant of Gemma 3 to use (e.g., "4b")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more diverse outputs)
        top_p: Nucleus sampling probability threshold
        top_k: Number of highest probability tokens to consider
        seed: Random seed for reproducibility
        multi_turn: Whether to enable multi-turn conversation mode
        
    Returns:
        Tuple of (generated_text, tokens_per_second) or (None, None) on failure
    """
    logger.info(f"Starting multimodal inference with prompt: '{prompt}'")
    
    # Check for image token in prompt
    if GemmaTokens.START_OF_IMAGE not in prompt:
        logger.warning(f"Prompt does not contain '{GemmaTokens.START_OF_IMAGE}'. Image may be ignored.")
    
    start_time = time.time()
    tokens_per_second = None
    
    try:
        # --- Load Model Components ---
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        loaded_params = gm.ckpts.load_params(path=checkpoint_path, text_only=False)
        params = ensure_consistent_dtypes(loaded_params)
        del loaded_params  # Free memory
        
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = gm.text.Gemma3Tokenizer(path=str(tokenizer_path))
        
        logger.info(f"Creating Gemma3_{model_size} model instance")
        if model_size.lower() == "4b":
            model = gm.nn.Gemma3_4B()
        elif model_size.lower() == "12b":
            model = gm.nn.Gemma3_12B() 
        elif model_size.lower() == "27b":
            model = gm.nn.Gemma3_27B()
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        logger.info(f"Creating ChatSampler (multi_turn={multi_turn})")
        sampler = gm.text.ChatSampler(
            model=model,
            params=params,
            tokenizer=tokenizer,
            multi_turn=multi_turn
        )
        
        # --- Process Image ---
        logger.info("Processing image...")
        image_processed = preprocess_image_for_inference(image_path)
        if image_processed is None:
            raise ValueError(f"Failed to process image: {image_path}")
        
        # --- Prepare Generation ---
        logger.info("Preparing for generation...")
        key = jax.random.PRNGKey(seed)
        
        # Calculate max_new_tokens if -1 (dynamic)
        if max_new_tokens == -1:
            model_context_len = get_model_context_length(model)
            input_ids = tokenizer.encode(prompt)
            prompt_len = len(input_ids)
            estimated_image_tokens = 256  # Heuristic for 896x896 image
            available_len = model_context_len - prompt_len - estimated_image_tokens
            actual_max_new_tokens = max(1, available_len - 20)  # Buffer
            logger.info(f"Dynamically set max_new_tokens: {actual_max_new_tokens}")
        else:
            actual_max_new_tokens = max(1, max_new_tokens)
        
        # --- Generate Output ---
        logger.info(f"Generating with max_new_tokens={actual_max_new_tokens}")
        gen_start_time = time.time()
        
        output_text = sampler.chat(
            prompt=prompt,
            images=image_processed,  # Pass the processed image (without batch dimension)
            max_new_tokens=actual_max_new_tokens,
            rng=key
        )
        
        gen_elapsed = time.time() - gen_start_time
        logger.info(f"Generation completed in {gen_elapsed:.2f}s")
        
        # --- Calculate Performance ---
        if not output_text:
            logger.warning("Model generated empty response")
            output_text = "[Model produced no output]"
        elif gen_elapsed > 0:
            tokens_per_second = actual_max_new_tokens / gen_elapsed
            logger.info(f"Performance: ~{tokens_per_second:.2f} tokens/second")
        
        # --- Analyze Output ---
        if 'extract_answer_from_response' in globals() and output_text:
            answer = extract_answer_from_response(output_text)
            if answer:
                logger.info(f"Extracted answer: '{answer}'")
            else:
                logger.info("No explicit answer found in response")
        
    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        return None, None
    finally:
        # Clean up resources to reduce memory usage
        jax.clear_caches()
        locals_to_delete = ['params', 'model', 'sampler', 'image_processed', 'tokenizer']
        for var in locals_to_delete:
            if var in locals():
                del locals()[var]
    
    total_elapsed = time.time() - start_time
    logger.info(f"Total inference time: {total_elapsed:.2f}s")
    return output_text, tokens_per_second

# --- Command Line Interface ---
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gemma 3 Multimodal Inference Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument("--question", type=str, help="Question text (alternative to --prompt)")
    parser.add_argument(
        "--prompt", type=str, 
        help="Full formatted prompt (overrides --question if both provided)"
    )
    parser.add_argument(
        "--image", type=str,
        help="Path to specific image file (overrides auto-detection)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_size", type=str, default="4b", 
        choices=["4b", "12b", "27b"],
        help="Gemma 3 model size"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="Maximum tokens to generate (-1 for dynamic)"
    )
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi_turn", action="store_true", help="Enable multi-turn conversation")
    
    # Path overrides
    parser.add_argument("--checkpoint_dir", type=str, help="Custom checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, help="Custom tokenizer path")
    parser.add_argument("--data_dir", type=str, help="Custom data directory")
    
    args = parser.parse_args()
    
    # Verify that at least one of question or prompt is provided
    if not args.question and not args.prompt:
        parser.error("Either --question or --prompt must be provided")
    
    return args

def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """
    Resolve all necessary file and directory paths.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing resolved paths
    """
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Project root: {project_root}")
    
    # Resolve checkpoint directory
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
    else:
        checkpoint_dir_name = f"gemma3-{args.model_size}"
        checkpoint_path = project_root / "pre-trained" / checkpoint_dir_name
    
    # Resolve tokenizer path
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
    else:
        tokenizer_path = project_root / "pre-trained" / "tokenizer.model"
    
    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data"
    
    images_dir = data_dir / "images"
    
    # Validate paths
    paths = {
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "images_dir": images_dir,
        "project_root": project_root
    }
    
    for name, path in paths.items():
        if name.endswith("_path"):
            if not path.exists():
                raise FileNotFoundError(f"{name.replace('_', ' ').title()} not found: {path}")
        elif name.endswith("_dir"):
            if not path.is_dir():
                raise NotADirectoryError(f"{name.replace('_', ' ').title()} not found: {path}")
    
    return paths

def find_image(paths: Dict[str, Path], image_path: Optional[str] = None) -> Path:
    """
    Find an image file to use for inference.
    
    Args:
        paths: Dictionary of resolved paths
        image_path: Optional specific image path from arguments
        
    Returns:
        Path to the image file
    """
    if image_path:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Specified image file not found: {image_file}")
        return image_file
    
    # Find all image files in the images directory
    images_dir = paths["images_dir"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {images_dir}")
    
    # Sort for consistency and take the first one
    image_files.sort()
    selected_image = image_files[0]
    logger.info(f"Selected image: {selected_image}")
    
    return selected_image

def main():
    """Main function: parse args, resolve paths, run inference."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Resolve paths
        paths = resolve_paths(args)
        
        # Find image to use
        image_path = find_image(paths, args.image)
        
        # Determine prompt to use
        if args.prompt:
            prompt = args.prompt
            logger.info("Using provided prompt")
        else:
            # Create a prompt using the question
            prompt = create_mathvista_prompt(
                question=args.question,
                include_image=True
            )
            logger.info("Created prompt from question")
        
        # Run inference
        result_text, tokens_per_sec = run_multimodal_inference(
            checkpoint_path=str(paths["checkpoint_path"]),
            tokenizer_path=str(paths["tokenizer_path"]),
            prompt=prompt,
            image_path=str(image_path),
            model_size=args.model_size,
            max_new_tokens=args.max_tokens,
            seed=args.seed,
            multi_turn=args.multi_turn
        )
        
        # Print results
        print("\n" + "="*70)
        if result_text:
            print("Multimodal Output:")
            print("-"*70)
            print(result_text)
            print("-"*70)
            if tokens_per_sec is not None:
                print(f"Performance: ~{tokens_per_sec:.2f} tokens/second")
        else:
            print("Multimodal Inference Failed")
            print("Please check logs for errors")
        print("="*70)
        
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"Path Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Ensure JAX caches are cleared
        jax.clear_caches()

if __name__ == "__main__":
    main()