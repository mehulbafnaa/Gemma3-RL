# src/model/gemma_vision.py

import os
import yaml
import logging
from pathlib import Path # Use pathlib for paths
import tensorflow as tf
import keras
from typing import Optional, List, Dict, Union # Added Union

# Attempt to import the official Gemma library
try:
    import gemma
    # Import specific Gemma components if needed
    from gemma import sampler # Example if using gemma library sampler
    # from gemma import keras as gemma_keras
    GEMMA_LIB_AVAILABLE = True
    # Define constants based on library if available
    IMAGE_TOKEN = getattr(gemma, 'IMAGE_TOKEN', '<img>') # Use library default if possible
    START_OF_TURN = getattr(gemma, 'START_OF_TURN', '<start_of_turn>')
    END_OF_TURN = getattr(gemma, 'END_OF_TURN', '<end_of_turn>')
    BOS_TOKEN = getattr(gemma, 'BOS_TOKEN', '<bos>')
    USER = getattr(gemma, 'USER', 'user')
    MODEL = getattr(gemma, 'MODEL', 'model')

except ImportError:
    logging.warning("Could not import official 'gemma' library. Model loading might rely solely on Keras/TF.")
    GEMMA_LIB_AVAILABLE = False
    gemma = None
    # Define fallback constants if library missing
    IMAGE_TOKEN = '<img>'
    START_OF_TURN = '<start_of_turn>'
    END_OF_TURN = '<end_of_turn>'
    BOS_TOKEN = '<bos>'
    USER = 'user'
    MODEL = 'model'


# SentencePiece needed ONLY if the gemma library *doesn't* bundle the tokenizer
# try:
#     import sentencepiece as spm
# except ImportError:
#     logging.error("SentencePiece library not found. Please install: pip install sentencepiece")
#     spm = None

# Setup logger for this module
logger = logging.getLogger(__name__)

# Placeholder for configuration loading
def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    config_path_obj = Path(config_path) # Work with Path objects
    if not config_path_obj.exists():
        logger.error(f"Configuration file not found: {config_path_obj}")
        return {}
    try:
        with open(config_path_obj, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path_obj}")
            return config if config else {}
    except Exception as e:
        logger.error(f"Error loading config {config_path_obj}: {e}", exc_info=True)
        return {}

class GemmaVisionModel(keras.Model):
    """
    Wrapper class for loading and interacting with a pre-trained Gemma 3 multimodal model.
    Assumes the model preset includes the tokenizer.
    """
    def __init__(self, config_path: str, **kwargs):
        """
        Initializes the model wrapper.

        Args:
            config_path (str): Path to the model configuration YAML file.
            **kwargs: Additional arguments passed to the keras.Model initializer.
        """
        super().__init__(**kwargs)
        self.config = load_config(config_path)

        # --- Determine Model Preset Path ---
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        default_preset_dir = "pre-trained/gemma3-4b"
        preset_relative_path = self.config.get("model", {}).get("preset_path", default_preset_dir)
        self.preset_path = str(self.project_root / preset_relative_path) # Keep as string for gemma lib? Check docs
        logger.info(f"Resolved model preset path to: {self.preset_path}")

        # --- Set Precision Policy ---
        try:
             policy_name = self.config.get("train", {}).get("mixed_precision_policy", "mixed_bfloat16")
             policy = keras.mixed_precision.Policy(policy_name)
             keras.mixed_precision.set_global_policy(policy)
             logger.info(f"Set Keras mixed precision policy to: {policy.name}")
        except Exception as e:
             logger.warning(f"Could not set mixed precision policy '{policy_name}': {e}. Using default.")

        # --- Load Pre-trained Model (which should include tokenizer) ---
        self.gemma_model = self._load_pretrained_model_with_tokenizer()

        # --- Get Tokenizer Reference ---
        self.tokenizer = None
        if self.gemma_model and hasattr(self.gemma_model, 'tokenizer'):
             self.tokenizer = self.gemma_model.tokenizer
             if self.tokenizer:
                  logger.info(f"Tokenizer successfully accessed from loaded Gemma model (Vocab size: {getattr(self.tokenizer, 'vocab_size', 'N/A')}).")
                  # Access token IDs if needed (names might vary, check Gemma 3 specifics)
                  self.pad_token_id = getattr(self.tokenizer, 'pad_id', 0) # Default pad ID often 0
                  # self.bos_token_id = getattr(self.tokenizer, 'bos_id', None) # Often handled by encode
                  self.eos_token_id = getattr(self.tokenizer, 'eos_id', 1) # Default EOS ID often 1
             else:
                  logger.error("Gemma model loaded, but tokenizer attribute is missing or None.")
        else:
            logger.error("Gemma model failed to load or does not have a 'tokenizer' attribute.")


    def _load_pretrained_model_with_tokenizer(self) -> Optional[keras.Model]: # Type hint might change based on actual model class
        """Loads the pre-trained Gemma model checkpoint, expecting it to bundle the tokenizer."""
        preset_path_obj = Path(self.preset_path)
        if not preset_path_obj.exists() or not preset_path_obj.is_dir():
             logger.error(f"Model preset directory not found or not a directory: {self.preset_path}")
             return None

        logger.info(f"Attempting to load Gemma 3 model and tokenizer from preset: {self.preset_path}")

        # --- !!! CRITICAL Placeholder - Needs Gemma 3 Specifics !!! ---
        try:
            if GEMMA_LIB_AVAILABLE and hasattr(gemma, 'Gemma') and hasattr(gemma.Gemma, 'from_preset'):
                 logger.info("Attempting load using gemma.Gemma.from_preset...")
                 # This function should handle finding tokenizer.model within the preset dir
                 model = gemma.Gemma.from_preset(self.preset_path)
                 logger.info("Model loaded successfully using gemma.Gemma.from_preset.")
                 # Verify tokenizer was loaded
                 if not hasattr(model, 'tokenizer') or not model.tokenizer:
                     logger.warning("gemma.Gemma.from_preset loaded model but tokenizer is missing!")
                 return model
            else:
                 logger.warning("Official 'gemma' library/method not available.")
                 raise NotImplementedError("Gemma library unavailable or from_preset missing.")

            # Add other loading options (Keras, TF Hub) here if needed,
            # but ensure they also somehow provide access to the correct tokenizer.

        except NotImplementedError:
             logger.warning("No suitable loading method found. Using a dummy placeholder.")
             # Fallback to a placeholder - REMOVE THIS IN REAL IMPLEMENTATION
             # ... (Dummy Keras model definition - same as before, but won't have .tokenizer) ...
             input_text = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_tokens")
             input_image = keras.layers.Input(shape=(896, 896, 3), dtype=tf.bfloat16, name="input_image") # Use known size
             text_emb = keras.layers.Embedding(262144, 1024)(input_text) # Match tokenizer size approx
             img_emb = keras.layers.Conv2D(1024, (16, 16), strides=(16,16))(input_image) # Dummy patch embedding
             img_emb = keras.layers.Reshape((-1, 1024))(img_emb) # Flatten patches
             # Dummy multimodal combination - extremely simplified
             combined = keras.layers.Concatenate(axis=1)([keras.layers.GlobalAveragePooling1D()(text_emb)[tf.newaxis, :], keras.layers.GlobalAveragePooling1D()(img_emb)])
             output = keras.layers.Dense(262144, name="dummy_output_logits")(combined) # Dummy output logits
             model = keras.Model(inputs=[input_text, input_image], outputs=output, name="DummyGemmaPlaceholder")
             logger.warning("Initialized a DUMMY model placeholder. Replace _load_pretrained_model_with_tokenizer with actual loading code.")
             return model # This dummy model won't have a .tokenizer attribute

        except Exception as e:
             logger.error(f"Failed to load Gemma 3 model from {self.preset_path}: {e}", exc_info=True)
             return None
        # --- !!! END Placeholder !!! ---

    def call(self, inputs, training: bool = False):
        """Defines the forward pass for the model."""
        # ... (Keep placeholder implementation as before, needs specifics) ...
        if not self.gemma_model:
            logger.error("Gemma model not loaded, cannot execute call.")
            return None
        try:
            logger.warning("GemmaVisionModel.call() is a placeholder, passing input to underlying model without verification.")
            return self.gemma_model(inputs, training=training)
        except Exception as e:
             logger.error(f"Error during GemmaVisionModel.call: {e}", exc_info=True)
             return None


    # def generate(self,
    #              prompts: Union[List[str], str],
    #              images: Optional[Union[tf.Tensor, List[Optional[tf.Tensor]]]] = None,
    #              max_length: int = 1024,
    #              temperature: float = 0.7,
    #              top_k: int = 40
    #              ) -> Union[List[str], str]:
    #     """Generates text based on text prompt(s) and optional image(s)."""
    #     # Use the tokenizer loaded by the model
    #     if not self.gemma_model or not self.tokenizer:
    #         logger.error("Model or its tokenizer not loaded correctly, cannot generate.")
    #         return ["Error: Model/Tokenizer not loaded."] * (len(prompts) if isinstance(prompts, list) else 1)

    #     is_batch = isinstance(prompts, list)
    #     if not is_batch:
    #         prompts = [prompts]
    #         images = [images]

    #     if images and len(prompts) != len(images):
    #          logger.error(f"Number of prompts ({len(prompts)}) does not match number of images ({len(images)}).")
    #          return ["Error: Prompt/Image mismatch."] * len(prompts)

    #     # --- 1. Prepare Formatted Prompts ---
    #     # Use constants defined earlier (either from gemma lib or fallbacks)
    #     formatted_prompts = []
    #     for prompt_content in prompts:
    #          prompt_content = str(prompt_content).strip()
    #          formatted = f"{BOS_TOKEN}{START_OF_TURN}{USER}\n{prompt_content}{END_OF_TURN}{START_OF_TURN}{MODEL}\n"
    #          formatted_prompts.append(formatted)

    #     # --- 2. Tokenize ---
    #     try:
    #         # Use the model's tokenizer
    #         # Assuming tokenizer.encode handles list input for batching if generate needs lists
    #         # Or maybe the generate method itself takes the raw formatted strings? Check docs.
    #         # Let's assume we tokenize first.
    #         tokenized_prompts = [self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in formatted_prompts]
    #         # Need to prepare inputs correctly for the underlying generate method
    #         # processed_inputs = tokenized_prompts # Or maybe padded tensors?
    #         processed_inputs = formatted_prompts # Many high-level generate methods take strings

    #     except Exception as e:
    #          logger.error(f"Error during prompt tokenization: {e}", exc_info=True)
    #          return [f"Error tokenizing prompt."] * len(prompts)

    #     # --- 3. Prepare Image Inputs ---
    #     # ... (Keep placeholder image processing logic as before) ...
    #     processed_images = None
    #     if images:
    #          try:
    #               valid_images = [img for img in images if img is not None]
    #               if valid_images:
    #                   processed_images = tf.stack(valid_images)
    #          except Exception as e:
    #               logger.error(f"Error processing image batch: {e}", exc_info=True)
    #               return [f"Error processing images."] * len(prompts)

    #     # --- 4. Generate Tokens ---
    #     logger.info(f"Starting generation for {len(prompts)} prompt(s)...")
    #     try:
    #         # --- !!! Placeholder for actual generation call !!! ---
    #         # This call needs to match the API of the loaded self.gemma_model
    #         # It might take strings, lists of tokens, image tensors/embeddings etc.

    #         # Example if using gemma library's generate
    #         # sampler_obj = sampler.TopKSampler(k=top_k, temperature=temperature) # Example
    #         # result_sequences = self.gemma_model.generate(
    #         #     processed_inputs, # Likely strings or token lists
    #         #     images=processed_images, # Hypothetical
    #         #     max_length=max_length,
    #         #     sampler=sampler_obj
    #         # ) # Returns list of token lists or strings? Check docs

    #         # --- Using a dummy response ---
    #         logger.warning("Using DUMMY generation response. Replace with actual model.generate() call.")
    #         # Simulate just returning the prompt + EOS id
    #         result_sequences = [[*p_tokens, self.eos_token_id] for p_tokens in tokenized_prompts]


    #     except Exception as e:
    #          logger.error(f"Error during model generation: {e}", exc_info=True)
    #          return [f"Error during generation."] * len(prompts)

    #     # --- 5. Decode Output ---
    #     generated_texts = []
    #     try:
    #         # Assuming result_sequences is a list of lists of token IDs
    #         for i, sequence in enumerate(result_sequences):
    #              prompt_len = len(tokenized_prompts[i])
    #              if len(sequence) > prompt_len:
    #                   new_tokens = sequence[prompt_len:]
    #              else:
    #                   new_tokens = [] # Assume nothing new generated if sequence not longer

    #              if hasattr(new_tokens, 'numpy'): new_tokens = new_tokens.numpy()
    #              if not isinstance(new_tokens, list): new_tokens = new_tokens.tolist()

    #              if new_tokens and new_tokens[-1] == self.eos_token_id:
    #                   new_tokens = new_tokens[:-1]

    #              decoded_text = self.tokenizer.decode(new_tokens)
    #              generated_texts.append(decoded_text.strip())

    #         logger.info(f"Generation completed for {len(prompts)} prompt(s).")

    #     except Exception as e:
    #         logger.error(f"Error during token decoding: {e}", exc_info=True)
    #         while len(generated_texts) < len(prompts):
    #              generated_texts.append("Error decoding output.")

    #     return generated_texts[0] if not is_batch else generated_texts

    def generate(self,
                    prompts: Union[List[str], str],
                    images: Optional[Union[tf.Tensor, List[Optional[tf.Tensor]]]] = None,
                    max_length: int = 1024,
                    temperature: float = 0.7, # Example sampling param
                    top_k: int = 40         # Example sampling param
                    ) -> Union[List[str], str]:
            """Generates text based on text prompt(s) and optional image(s)."""
            # Use the tokenizer loaded by the model
            if not self.gemma_model or not self.tokenizer:
                logger.error("Model or its tokenizer not loaded correctly, cannot generate.")
                return ["Error: Model/Tokenizer not loaded."] * (len(prompts) if isinstance(prompts, list) else 1)

            is_batch = isinstance(prompts, list)
            if not is_batch:
                prompts = [prompts]
                images = [images] # Wrap single image/None in a list

            if images and len(prompts) != len(images):
                logger.error(f"Number of prompts ({len(prompts)}) does not match number of images ({len(images)}).")
                return ["Error: Prompt/Image mismatch."] * len(prompts)

            # --- 1. Prepare Formatted Prompts for Model ---
            formatted_prompts = []
            for prompt_content in prompts:
                prompt_content = str(prompt_content).strip()
                formatted = f"{BOS_TOKEN}{START_OF_TURN}{USER}\n{prompt_content}{END_OF_TURN}{START_OF_TURN}{MODEL}\n"
                formatted_prompts.append(formatted)

            # --- 2. Tokenize (Potentially needed for decoding later or if model takes tokens) ---
            try:
                # We still need the tokenized version to calculate prompt length for decoding
                tokenized_prompts = [self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in formatted_prompts]
            except Exception as e:
                logger.error(f"Error during prompt tokenization: {e}", exc_info=True)
                return [f"Error tokenizing prompt."] * len(prompts)

            # --- 3. Prepare Image Inputs (if any) ---
            processed_images = None
            if images:
                try:
                    valid_images = [img for img in images if img is not None]
                    if valid_images:
                        processed_images = tf.stack(valid_images) # Assumes all are tensors of same shape
                except Exception as e:
                    logger.error(f"Error processing image batch: {e}", exc_info=True)
                    return [f"Error processing images."] * len(prompts)

            # --- 4. Generate Tokens ---
            logger.info(f"Starting generation for {len(prompts)} prompt(s)...")
            try:
                # --- !!! CORRECTED: Call the underlying model's generate method !!! ---
                # Ensure arguments match how the mock expects to be called, OR
                # how the real Gemma 3 generate method works.
                # Passing formatted_prompts (list of strings) here, as the mock handles it.
                result_sequences = self.gemma_model.generate(
                    formatted_prompts, # Pass formatted strings
                    images=processed_images,
                    max_length=max_length,
                    # Add other necessary args like temperature, top_k if generate accepts them directly
                    # temperature=temperature,
                    # top_k=top_k
                )
                # --- !!! END CORRECTION !!! ---

            except Exception as e:
                logger.error(f"Error during model generation: {e}", exc_info=True)
                return [f"Error during generation."] * len(prompts)

            # --- 5. Decode Output ---
            # (Decoding logic remains the same)
            generated_texts = []
            try:
                # Assuming result_sequences is a list of lists of token IDs (mock returns this)
                for i, sequence in enumerate(result_sequences):
                    prompt_len = len(tokenized_prompts[i])
                    if len(sequence) > prompt_len:
                        new_tokens = sequence[prompt_len:]
                    else:
                        new_tokens = []

                    if hasattr(new_tokens, 'numpy'): new_tokens = new_tokens.numpy()
                    if not isinstance(new_tokens, list): new_tokens = new_tokens.tolist()

                    # Use self.eos_token_id if available, otherwise fallback
                    eos_id_to_check = getattr(self, 'eos_token_id', 1) # Default to 1 if not set
                    if new_tokens and new_tokens[-1] == eos_id_to_check:
                        new_tokens = new_tokens[:-1]

                    decoded_text = self.tokenizer.decode(new_tokens)
                    generated_texts.append(decoded_text.strip())

                logger.info(f"Generation completed for {len(prompts)} prompt(s).")

            except Exception as e:
                logger.error(f"Error during token decoding: {e}", exc_info=True)
                while len(generated_texts) < len(prompts):
                    generated_texts.append("Error decoding output.")

            return generated_texts[0] if not is_batch else generated_texts

# --- Example Usage (for direct testing) ---
if __name__ == '__main__':
    print("Running GemmaVisionModel example usage...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    current_dir = Path(__file__).parent
    project_root_path = current_dir.parent.parent
    cfg_path = project_root_path / "configs/gemma3_4B_config.yaml"

    print(f"Project Root: {project_root_path}")
    print(f"Config Path: {cfg_path}")

    if not cfg_path.exists():
        print(f"ERROR: Config file not found at {cfg_path}")

    # --- Load the model (no tokenizer path needed) ---
    model_wrapper = GemmaVisionModel(config_path=str(cfg_path))

    # --- Test generation (only if model and its tokenizer loaded) ---
    if model_wrapper.gemma_model and model_wrapper.tokenizer:
        test_image_tensor = None # Placeholder
        prompt1 = "Explain the concept of KL divergence simply."
        prompt2 = f"{IMAGE_TOKEN}\nWhat objects are in this image?" if test_image_tensor is not None else "What is the capital of France?"
        prompts_to_test = [prompt1, prompt2]
        images_to_test = [None, test_image_tensor]

        print(f"\n--- Running Generation Test ---")
        outputs = model_wrapper.generate(
            prompts=prompts_to_test,
            images=images_to_test,
            max_length=64
        )

        print("\n--- Generated Outputs ---")
        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts_to_test[i]}")
            print(f"Output {i+1}: {output}\n")
    else:
        print("\nSkipping generation test as model or tokenizer failed to load correctly.")