import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configure logging - Set level to DEBUG to see detailed parameter processing logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# To enable debug logs, uncomment the line below:
# logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    import jax
    import jax.numpy as jnp
    # Flax traverse_util might not be needed anymore if process_parameters is removed
    # import flax.traverse_util
    # Orbax might not be needed directly if using gm.ckpts.load_params
    # import orbax.checkpoint
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
except ImportError as e:
    logger.error(f"Error importing core JAX libraries: {e}")
    logger.error("Please ensure JAX is installed.")
    sys.exit(1)

# Try importing the specific gemma library
try:
    from gemma import gm
    logger.info("Successfully imported 'gemma' library.")
except ImportError as e:
    logger.error(f"Error importing the 'gemma' library: {e}")
    logger.error("Please ensure the 'gemma' library is installed and accessible in your PYTHONPATH.")
    sys.exit(1)
except AttributeError as e:
    logger.error(f"Error accessing components within the 'gemma' library: {e}")
    logger.error("The 'gemma' library structure might have changed or is incomplete.")
    sys.exit(1)


# REMOVED: unwrap_w_dictionaries function

def ensure_consistent_dtypes(params: Any, target_dtype=jnp.float32) -> Any:
    """
    Ensure all floating-point arrays have consistent data types.
    Defaults to float32 for better compatibility, bfloat16 can be used if needed.
    Applying this AFTER gm.ckpts.load_params as a safety measure.

    Args:
        params: Parameter tree (potentially loaded by gm.ckpts.load_params)
        target_dtype: Target data type for floating-point arrays (default: jnp.float32)

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

# REMOVED: process_parameters function, as gm.ckpts.load_params should handle structure.
# def process_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
#    ...

# Renamed max_tokens -> max_new_tokens for clarity
def run_text_inference(checkpoint_path: str, tokenizer_path: str, prompt: str, max_new_tokens: int = 100, seed: int = 42) -> Tuple[Optional[str], Optional[float]]:
    """
    Run text-only inference with Gemma3 model using standard JIT compilation.
    Uses gm.ckpts.load_params for potentially simpler and more robust loading.
    Calculates and returns tokens per second. Tries to dynamically get context length.
    Prints the model architecture after instantiation.

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
    logger.info(f"Starting text-only inference for prompt: '{prompt}'")
    tokens_per_second = None # Initialize tok/sec
    model_context_length = None # Initialize context length

    # Step 1: Load parameters using gm.ckpts.load_params
    try:
        logger.info(f"Loading checkpoint using gm.ckpts.load_params from: {checkpoint_path}")
        start_time = time.time()

        # Use the library's loading function
        # Set text_only=True as we don't need vision parameters
        loaded_params = gm.ckpts.load_params(
            path=checkpoint_path,
            text_only=True,
            # Add sharding=... if you have specific sharding requirements
            # Add quantize=True if needed
        )

        elapsed = time.time() - start_time
        logger.info(f"Checkpoint loaded via gm.ckpts.load_params in {elapsed:.2f} seconds")

        # Optional Step: Ensure consistent dtypes after loading (safety check)
        logger.info("Ensuring consistent data types (target=float32)...")
        processed_params = ensure_consistent_dtypes(loaded_params, target_dtype=jnp.float32)
        # Use jnp.bfloat16 if needed and supported:
        # processed_params = ensure_consistent_dtypes(loaded_params, target_dtype=jnp.bfloat16)
        logger.info("Data types checked/adjusted.")

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
        # Check if the error relates to the checkpoint format itself
        if "is not a valid Orbax checkpoint" in str(e):
             logger.error("The specified path might not be a compatible Orbax checkpoint.")
        return None, None

    # Step 2: Load tokenizer (remains the same)
    tokenizer = None # Initialize tokenizer variable
    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = gm.text.Gemma3Tokenizer(path=str(tokenizer_path))
        logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.exception(f"Fatal error loading tokenizer: {e}", exc_info=True)
        return None, None

    # Step 3: Create model instance, print architecture, and try to get context length
    model = None # Initialize model variable
    try:
        logger.info("Creating Gemma3_4B model instance")
        if not hasattr(gm, 'nn') or not hasattr(gm.nn, 'Gemma3_4B'):
             logger.error("Model class 'gm.nn.Gemma3_4B' not found in the imported library.")
             return None, None
        model = gm.nn.Gemma3_4B()
        logger.info(f"Model instance created: {model.__class__.__name__}")

        # --- Print Model Architecture ---
        # Printing the model object usually gives its string representation (architecture)
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE:")
        print("-"*60)
        print(model) # Print the instantiated model object
        print("="*60 + "\n")
        # --- End of Model Architecture Print ---


        # --- Attempt to get context length dynamically ---
        # Try common attribute names. Provide a default (e.g., 8192 or 128k for Gemma3 4B)
        # Based on search results, 128k is likely for Gemma3 4B
        default_context_length = 128000 # Use 128k as default for Gemma3 4B
        context_length_attr = None
        possible_attr_names = ['context_length', 'max_position_embeddings', 'seq_length']

        if hasattr(model, 'config'): # Check if there's a config object first
             for attr_name in possible_attr_names:
                 context_length_attr = getattr(model.config, attr_name, None)
                 if context_length_attr is not None:
                     logger.info(f"Found context length attribute in model.config.{attr_name}")
                     break

        if context_length_attr is None: # If not found in config, check model directly
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
        # --- End of context length attempt ---

    except Exception as e:
        logger.exception(f"Fatal error creating model instance: {e}", exc_info=True)
        return None, None

    # Step 4: Create sampler instance (remains the same, uses processed_params)
    sampler = None # Initialize sampler variable
    try:
        logger.info("Creating text sampler instance")
        if not hasattr(gm, 'text') or not hasattr(gm.text, 'Sampler'):
            logger.error("Sampler class 'gm.text.Sampler' not found in the imported library.")
            return None, None
        # Pass the parameters loaded by gm.ckpts.load_params
        sampler = gm.text.Sampler(
            model=model,
            params=processed_params,
            tokenizer=tokenizer # Pass the loaded tokenizer
        )
        logger.info("Sampler instance created successfully")
    except Exception as e:
        logger.exception(f"Fatal error creating sampler instance: {e}", exc_info=True)
        if "missing key" in str(e).lower() or "structure mismatch" in str(e).lower():
            logger.error("This error often indicates a mismatch between loaded parameter names/structure and what the model/sampler expects.")
            logger.error("Even with gm.ckpts.load_params, the checkpoint might not perfectly match the model definition.")
        return None, None

    # Step 5: Run text generation and calculate tokens/sec
    output_text = None
    try:
        # Determine the actual max_new_tokens to use
        actual_max_new_tokens = max_new_tokens
        if max_new_tokens == -1: # Use -1 as a flag to use full context
             if model_context_length:
                 # Calculate remaining tokens, ensuring it's not negative
                 input_ids_for_len = tokenizer.encode(prompt)
                 prompt_len = len(input_ids_for_len)
                 actual_max_new_tokens = max(1, model_context_length - prompt_len - 10) # Subtract prompt len and a buffer
                 logger.info(f"Using calculated max_new_tokens based on context length: {actual_max_new_tokens}")
             else:
                 actual_max_new_tokens = 2048 # Fallback if context length couldn't be determined
                 logger.warning(f"max_new_tokens was -1 but context length unknown. Using fallback: {actual_max_new_tokens}")
        elif max_new_tokens <= 0:
             logger.warning(f"max_new_tokens ({max_new_tokens}) is invalid. Using default 100.")
             actual_max_new_tokens = 100


        logger.info(f"Generating text with max_new_tokens={actual_max_new_tokens}...")
        key = jax.random.PRNGKey(seed)

        # Tokenize the input prompt to get the starting token count
        input_ids = tokenizer.encode(prompt)
        num_input_tokens = len(input_ids)
        logger.info(f"Number of input tokens: {num_input_tokens}")

        with jax.default_matmul_precision('float32'):
            start_gen_time = time.time()
            output_text = sampler.sample( # Store result in output_text
                prompt=prompt,
                max_new_tokens=actual_max_new_tokens, # Use the determined value
                rng=key
            )
            # Ensure generation stops if max_tokens is reached or EOS is generated
            # The sampler should handle this internally based on max_new_tokens
            gen_elapsed = time.time() - start_gen_time
            logger.info(f"Text generation finished in {gen_elapsed:.2f} seconds.")

        # Ensure output is a string
        if not isinstance(output_text, str):
             logger.warning(f"Sampler output type is not string: {type(output_text)}. Attempting conversion.")
             output_text = str(output_text)

        # Calculate tokens per second
        output_ids = tokenizer.encode(output_text)
        num_output_tokens = len(output_ids)
        # Calculate generated tokens based on the difference from the prompt
        num_generated_tokens = num_output_tokens - num_input_tokens

        logger.info(f"Number of output tokens: {num_output_tokens}")
        logger.info(f"Number of generated tokens: {num_generated_tokens}")

        # Ensure num_generated_tokens isn't negative (e.g., if output is shorter than prompt somehow)
        num_generated_tokens = max(0, num_generated_tokens)

        if gen_elapsed > 0 and num_generated_tokens > 0:
            tokens_per_second = num_generated_tokens / gen_elapsed
            logger.info(f"Tokens per second: {tokens_per_second:.2f}")
        elif num_generated_tokens <= 0:
             logger.warning("No new tokens were generated.")
        else:
            logger.warning("Generation time was zero or negative, cannot calculate tokens per second.")

        return output_text, tokens_per_second # Return both results

    except Exception as e:
        logger.exception(f"Fatal error during text generation: {e}", exc_info=True)
        if "cannot determine the shape of {'w'" in str(e).lower():
            logger.error("JIT compilation failed: Cannot determine shape of {'w': ...} dictionary.")
            logger.error("This suggests an issue within the 'gemma' library's layers or how gm.ckpts.load_params prepares the structure.")
        elif "out of memory" in str(e).lower() or "resourceexhaustederror" in str(e).lower():
             logger.error("Out of memory error during generation.")
        elif 'scopeparamnotfounderror' in str(e).lower():
             logger.error("ScopeParamNotFoundError occurred. Indicates parameter names/structure loaded by gm.ckpts.load_params still don't match model expectations.")

        return None, None # Return None for both on error

def main():
    # Get script directory to base paths on
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent # Assumes tests/ is one level down from root

    # Define paths relative to the project root
    checkpoint_path = project_root / "pre-trained" / "gemma3-4b"
    tokenizer_path = project_root / "pre-trained" / "tokenizer.model"

    # Log and validate paths
    logger.info(f"Using Project Root: {project_root}")
    logger.info(f"Using Checkpoint Path: {checkpoint_path}")
    logger.info(f"Using Tokenizer Path: {tokenizer_path}")

    # Validate paths (gm.ckpts.load_params might do its own checks too)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logger.error(f"Checkpoint directory not found or not a directory: {checkpoint_path}")
        logger.error("Please ensure the checkpoint is correctly downloaded and placed.")
        sys.exit(1)

    if not tokenizer_path.exists() or not tokenizer_path.is_file():
        logger.error(f"Tokenizer file not found or not a file: {tokenizer_path}")
        logger.error("Please ensure the tokenizer.model file exists.")
        sys.exit(1)

    # Define the prompt
    prompt = "Explain the concept of Large Language Models in simple terms."

    # --- Set max_new_tokens ---
    # Set to a specific number of new tokens to generate (e.g., 512).
    # Or set to -1 to attempt using the model's full context length (calculated dynamically).
    # Using full context length can be very slow and memory intensive.
    # max_new_tokens_to_generate = 512
    max_new_tokens_to_generate = -1 # Use -1 flag to try dynamic calculation based on context length
    logger.info(f"Requested max_new_tokens: {max_new_tokens_to_generate} (-1 means use dynamic context length)")


    # Run the inference
    result_text, tok_sec = run_text_inference( # Capture both return values
        checkpoint_path=str(checkpoint_path), # Ensure paths are strings
        tokenizer_path=str(tokenizer_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens_to_generate # Pass the requested value
    )

    # Print the result
    print("\n" + "="*60)
    if result_text:
        print("TEXT-ONLY OUTPUT:")
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
        print("Common issues include parameter loading/processing errors,")
        print("missing dependencies, incorrect paths, or insufficient memory.")
    print("="*60)

if __name__ == "__main__":
    main()

