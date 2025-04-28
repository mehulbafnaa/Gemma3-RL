import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required libraries
try:
    import jax
    import jax.numpy as jnp
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
except ImportError as e:
    logger.error(f"Error importing core JAX libraries: {e}")
    sys.exit(1)

# Try importing the specific gemma library
try:
    from gemma import gm
    logger.info("Successfully imported 'gemma' library.")
except ImportError as e:
    logger.error(f"Error importing the 'gemma' library: {e}")
    sys.exit(1)

def ensure_consistent_dtypes(params: Any, target_dtype=jnp.bfloat16) -> Any:
    """
    Ensure all floating-point arrays have consistent data types.
    Changed default to bfloat16 for TPU optimization.

    Args:
        params: Parameter tree (potentially loaded by gm.ckpts.load_params)
        target_dtype: Target data type for floating-point arrays (default: jnp.bfloat16)

    Returns:
        Parameters with consistent data types
    """
    if isinstance(params, dict):
        return {k: ensure_consistent_dtypes(v, target_dtype) for k, v in params.items()}
    elif isinstance(params, (list, tuple)):
        return type(params)(ensure_consistent_dtypes(v, target_dtype) for v in params)
    elif hasattr(params, 'dtype') and hasattr(params, 'astype'):
        # Convert only floating-point arrays to target dtype
        if jnp.issubdtype(params.dtype, jnp.floating):
            if params.dtype != target_dtype:
                logger.debug(f"Converting array from {params.dtype} to {target_dtype}")
                return jnp.asarray(params, dtype=target_dtype)
            else:
                 logger.debug(f"Array already has target dtype {target_dtype}")
                 return params # Already correct type
        # Keep non-floating point arrays as they are
        logger.debug(f"Keeping non-floating array with dtype {params.dtype}")
        return params
    # Return non-array elements as is
    return params

def run_text_inference(checkpoint_path: str, tokenizer_path: str, prompt: str, max_new_tokens: int = 100, seed: int = 42) -> Tuple[Optional[str], Optional[float]]:
    """
    Run text-only inference with Gemma3 model using bfloat16 precision.
    
    Args:
        checkpoint_path: Path to the Orbax checkpoint directory
        tokenizer_path: Path to the tokenizer model file
        prompt: Text prompt
        max_new_tokens: Maximum **new** tokens to generate. If set to -1, will attempt
                        to use the model's full context length.
        seed: Random seed

    Returns:
        A tuple containing:
            - Generated text (str) or None if an error occurs
            - Tokens per second (float) or None if an error occurs or generation time is zero.
    """
    logger.info(f"Starting text-only inference with bfloat16 precision for prompt: '{prompt}'")
    tokens_per_second = None
    model_context_length = None

    # Step 1: Load parameters using gm.ckpts.load_params
    try:
        logger.info(f"Loading checkpoint using gm.ckpts.load_params from: {checkpoint_path}")
        start_time = time.time()

        loaded_params = gm.ckpts.load_params(
            path=checkpoint_path,
            text_only=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"Checkpoint loaded via gm.ckpts.load_params in {elapsed:.2f} seconds")

        # Using bfloat16 precision for better TPU performance
        logger.info("Converting parameters to bfloat16 precision...")
        processed_params = ensure_consistent_dtypes(loaded_params, target_dtype=jnp.bfloat16)
        logger.info("Parameters converted to bfloat16.")

        # Log top-level keys for verification
        if isinstance(processed_params, dict):
             logger.debug(f"Top-level parameter keys after loading: {list(processed_params.keys())}")
        else:
             logger.warning("Loaded parameters are not a dictionary.")

    except FileNotFoundError:
        logger.error(f"Checkpoint path not found by gm.ckpts.load_params: {checkpoint_path}")
        return None, None
    except Exception as e:
        logger.exception(f"Fatal error loading parameters using gm.ckpts.load_params: {e}", exc_info=True)
        if "is not a valid Orbax checkpoint" in str(e):
             logger.error("The specified path might not be a compatible Orbax checkpoint.")
        return None, None

    # Step 2: Load tokenizer (remains the same)
    tokenizer = None
    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = gm.text.Gemma3Tokenizer(path=str(tokenizer_path))
        logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.exception(f"Fatal error loading tokenizer: {e}", exc_info=True)
        return None, None

    # Step 3: Create model instance
    model = None
    try:
        logger.info("Creating Gemma3_4B model instance")
        if not hasattr(gm, 'nn') or not hasattr(gm.nn, 'Gemma3_4B'):
             logger.error("Model class 'gm.nn.Gemma3_4B' not found in the imported library.")
             return None, None
        model = gm.nn.Gemma3_4B()
        logger.info(f"Model instance created: {model.__class__.__name__}")

        # Print Model Architecture
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE:")
        print("-"*60)
        print(model)
        print("="*60 + "\n")

        # Attempt to get context length dynamically
        default_context_length = 128000
        context_length_attr = None
        possible_attr_names = ['context_length', 'max_position_embeddings', 'seq_length']

        if hasattr(model, 'config'):
             for attr_name in possible_attr_names:
                 context_length_attr = getattr(model.config, attr_name, None)
                 if context_length_attr is not None:
                     logger.info(f"Found context length attribute in model.config.{attr_name}")
                     break

        if context_length_attr is None:
             for attr_name in possible_attr_names:
                 context_length_attr = getattr(model, attr_name, None)
                 if context_length_attr is not None:
                     logger.info(f"Found context length attribute in model.{attr_name}")
                     break

        if isinstance(context_length_attr, int) and context_length_attr > 0:
             model_context_length = context_length_attr
             logger.info(f"Dynamically determined model context length: {model_context_length}")
        else:
             model_context_length = default_context_length
             logger.warning(f"Could not dynamically determine context length. Using default: {model_context_length}")

    except Exception as e:
        logger.exception(f"Fatal error creating model instance: {e}", exc_info=True)
        return None, None

    # Step 4: Create sampler instance
    sampler = None
    try:
        logger.info("Creating text sampler instance")
        if not hasattr(gm, 'text') or not hasattr(gm.text, 'Sampler'):
            logger.error("Sampler class 'gm.text.Sampler' not found in the imported library.")
            return None, None
        sampler = gm.text.Sampler(
            model=model,
            params=processed_params,
            tokenizer=tokenizer
        )
        logger.info("Sampler instance created successfully")
    except Exception as e:
        logger.exception(f"Fatal error creating sampler instance: {e}", exc_info=True)
        if "missing key" in str(e).lower() or "structure mismatch" in str(e).lower():
            logger.error("This error often indicates a mismatch between loaded parameter names/structure and what the model/sampler expects.")
        return None, None

    # Step 5: Run text generation with bfloat16 precision
    output_text = None
    try:
        # Determine the actual max_new_tokens to use
        actual_max_new_tokens = max_new_tokens
        if max_new_tokens == -1:
             if model_context_length:
                 input_ids_for_len = tokenizer.encode(prompt)
                 prompt_len = len(input_ids_for_len)
                 actual_max_new_tokens = max(1, model_context_length - prompt_len - 10)
                 logger.info(f"Using calculated max_new_tokens based on context length: {actual_max_new_tokens}")
             else:
                 actual_max_new_tokens = 2048
                 logger.warning(f"max_new_tokens was -1 but context length unknown. Using fallback: {actual_max_new_tokens}")
        elif max_new_tokens <= 0:
             logger.warning(f"max_new_tokens ({max_new_tokens}) is invalid. Using default 100.")
             actual_max_new_tokens = 100

        logger.info(f"Generating text with max_new_tokens={actual_max_new_tokens} using bfloat16 precision...")
        key = jax.random.PRNGKey(seed)

        # Tokenize the input prompt to get the starting token count
        input_ids = tokenizer.encode(prompt)
        num_input_tokens = len(input_ids)
        logger.info(f"Number of input tokens: {num_input_tokens}")

        # Set to bfloat16 precision for generation
        with jax.default_matmul_precision('bfloat16'):
            start_gen_time = time.time()
            output_text = sampler.sample(
                prompt=prompt,
                max_new_tokens=actual_max_new_tokens,
                rng=key
            )
            gen_elapsed = time.time() - start_gen_time
            logger.info(f"Text generation finished in {gen_elapsed:.2f} seconds.")

        # Ensure output is a string
        if not isinstance(output_text, str):
             logger.warning(f"Sampler output type is not string: {type(output_text)}. Attempting conversion.")
             output_text = str(output_text)

        # Calculate tokens per second
        output_ids = tokenizer.encode(output_text)
        num_output_tokens = len(output_ids)
        num_generated_tokens = num_output_tokens - num_input_tokens
        num_generated_tokens = max(0, num_generated_tokens)

        logger.info(f"Number of output tokens: {num_output_tokens}")
        logger.info(f"Number of generated tokens: {num_generated_tokens}")

        if gen_elapsed > 0 and num_generated_tokens > 0:
            tokens_per_second = num_generated_tokens / gen_elapsed
            logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        elif num_generated_tokens <= 0:
             logger.warning("No new tokens were generated.")
        else:
            logger.warning("Generation time was zero or negative, cannot calculate tokens per second.")

        return output_text, tokens_per_second

    except Exception as e:
        logger.exception(f"Fatal error during text generation: {e}", exc_info=True)
        if "out of memory" in str(e).lower() or "resourceexhaustederror" in str(e).lower():
             logger.error("Out of memory error during generation.")
        elif 'scopeparamnotfounderror' in str(e).lower():
             logger.error("ScopeParamNotFoundError occurred. Indicates parameter names/structure loaded by gm.ckpts.load_params still don't match model expectations.")

        return None, None

def main():
    # Get script directory to base paths on
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Define paths relative to the project root
    checkpoint_path = project_root / "pre-trained" / "gemma3-4b"
    tokenizer_path = project_root / "pre-trained" / "tokenizer.model"

    # Log and validate paths
    logger.info(f"Using Project Root: {project_root}")
    logger.info(f"Using Checkpoint Path: {checkpoint_path}")
    logger.info(f"Using Tokenizer Path: {tokenizer_path}")

    # Validate paths
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logger.error(f"Checkpoint directory not found or not a directory: {checkpoint_path}")
        sys.exit(1)

    if not tokenizer_path.exists() or not tokenizer_path.is_file():
        logger.error(f"Tokenizer file not found or not a file: {tokenizer_path}")
        sys.exit(1)

    # Define the prompt
    prompt = "Explain the concept of Large Language Models in simple terms."

    # Set max_new_tokens
    max_new_tokens_to_generate = -1
    logger.info(f"Requested max_new_tokens: {max_new_tokens_to_generate} (-1 means use dynamic context length)")

    # Configure JAX to use bfloat16 precision by default
    jax.config.update('jax_default_matmul_precision', 'bfloat16')
    logger.info("JAX configured to use bfloat16 precision by default")

    # Run the inference with bfloat16 precision
    result_text, tok_sec = run_text_inference(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens_to_generate
    )

    # Print the result
    print("\n" + "="*60)
    if result_text:
        print("TEXT-ONLY OUTPUT (BFLOAT16 PRECISION):")
        print("-"*60)
        print(result_text)
        print("-"*60)
        if tok_sec is not None:
            print(f"Performance: {tok_sec:.2f} tokens/second")
        else:
            print("Performance: Could not calculate tokens/second.")
    else:
        print("INFERENCE FAILED:")
        print("-"*60)
        print("Could not generate text. Please check the logs above for errors.")
    print("="*60)

if __name__ == "__main__":
    main()