# # src/model/gemma_vision.py

# import os
# import yaml
# import logging
# from pathlib import Path # Use pathlib for paths
# import tensorflow as tf
# import keras
# from typing import Optional, List, Dict, Union # Added Union

# # Attempt to import the official Gemma library
# try:
#     import gemma
#     # Import specific Gemma components if needed
#     from gemma import sampler # Example if using gemma library sampler
#     # from gemma import keras as gemma_keras
#     GEMMA_LIB_AVAILABLE = True
#     # Define constants based on library if available
#     IMAGE_TOKEN = getattr(gemma, 'IMAGE_TOKEN', '<img>') # Use library default if possible
#     START_OF_TURN = getattr(gemma, 'START_OF_TURN', '<start_of_turn>')
#     END_OF_TURN = getattr(gemma, 'END_OF_TURN', '<end_of_turn>')
#     BOS_TOKEN = getattr(gemma, 'BOS_TOKEN', '<bos>')
#     USER = getattr(gemma, 'USER', 'user')
#     MODEL = getattr(gemma, 'MODEL', 'model')

# except ImportError:
#     logging.warning("Could not import official 'gemma' library. Model loading might rely solely on Keras/TF.")
#     GEMMA_LIB_AVAILABLE = False
#     gemma = None
#     # Define fallback constants if library missing
#     IMAGE_TOKEN = '<img>'
#     START_OF_TURN = '<start_of_turn>'
#     END_OF_TURN = '<end_of_turn>'
#     BOS_TOKEN = '<bos>'
#     USER = 'user'
#     MODEL = 'model'


# # SentencePiece needed ONLY if the gemma library *doesn't* bundle the tokenizer
# # try:
# #     import sentencepiece as spm
# # except ImportError:
# #     logging.error("SentencePiece library not found. Please install: pip install sentencepiece")
# #     spm = None

# # Setup logger for this module
# logger = logging.getLogger(__name__)

# # Placeholder for configuration loading
# def load_config(config_path: str) -> dict:
#     """Loads configuration from a YAML file."""
#     config_path_obj = Path(config_path) # Work with Path objects
#     if not config_path_obj.exists():
#         logger.error(f"Configuration file not found: {config_path_obj}")
#         return {}
#     try:
#         with open(config_path_obj, 'r') as f:
#             config = yaml.safe_load(f)
#             logger.info(f"Loaded configuration from {config_path_obj}")
#             return config if config else {}
#     except Exception as e:
#         logger.error(f"Error loading config {config_path_obj}: {e}", exc_info=True)
#         return {}

# class GemmaVisionModel(keras.Model):
#     """
#     Wrapper class for loading and interacting with a pre-trained Gemma 3 multimodal model.
#     Assumes the model preset includes the tokenizer.
#     """
#     def __init__(self, config_path: str, **kwargs):
#         """
#         Initializes the model wrapper.

#         Args:
#             config_path (str): Path to the model configuration YAML file.
#             **kwargs: Additional arguments passed to the keras.Model initializer.
#         """
#         super().__init__(**kwargs)
#         self.config = load_config(config_path)

#         # --- Determine Model Preset Path ---
#         self.project_root = Path(__file__).parent.parent.parent.resolve()
#         default_preset_dir = "pre-trained/gemma3-4b"
#         preset_relative_path = self.config.get("model", {}).get("preset_path", default_preset_dir)
#         self.preset_path = str(self.project_root / preset_relative_path) # Keep as string for gemma lib? Check docs
#         logger.info(f"Resolved model preset path to: {self.preset_path}")

#         # --- Set Precision Policy ---
#         try:
#              policy_name = self.config.get("train", {}).get("mixed_precision_policy", "mixed_bfloat16")
#              policy = keras.mixed_precision.Policy(policy_name)
#              keras.mixed_precision.set_global_policy(policy)
#              logger.info(f"Set Keras mixed precision policy to: {policy.name}")
#         except Exception as e:
#              logger.warning(f"Could not set mixed precision policy '{policy_name}': {e}. Using default.")

#         # --- Load Pre-trained Model (which should include tokenizer) ---
#         self.gemma_model = self._load_pretrained_model_with_tokenizer()

#         # --- Get Tokenizer Reference ---
#         self.tokenizer = None
#         if self.gemma_model and hasattr(self.gemma_model, 'tokenizer'):
#              self.tokenizer = self.gemma_model.tokenizer
#              if self.tokenizer:
#                   logger.info(f"Tokenizer successfully accessed from loaded Gemma model (Vocab size: {getattr(self.tokenizer, 'vocab_size', 'N/A')}).")
#                   # Access token IDs if needed (names might vary, check Gemma 3 specifics)
#                   self.pad_token_id = getattr(self.tokenizer, 'pad_id', 0) # Default pad ID often 0
#                   # self.bos_token_id = getattr(self.tokenizer, 'bos_id', None) # Often handled by encode
#                   self.eos_token_id = getattr(self.tokenizer, 'eos_id', 1) # Default EOS ID often 1
#              else:
#                   logger.error("Gemma model loaded, but tokenizer attribute is missing or None.")
#         else:
#             logger.error("Gemma model failed to load or does not have a 'tokenizer' attribute.")


#     def _load_pretrained_model_with_tokenizer(self) -> Optional[keras.Model]: # Type hint might change based on actual model class
#         """Loads the pre-trained Gemma model checkpoint, expecting it to bundle the tokenizer."""
#         preset_path_obj = Path(self.preset_path)
#         if not preset_path_obj.exists() or not preset_path_obj.is_dir():
#              logger.error(f"Model preset directory not found or not a directory: {self.preset_path}")
#              return None

#         logger.info(f"Attempting to load Gemma 3 model and tokenizer from preset: {self.preset_path}")

#         # --- !!! CRITICAL Placeholder - Needs Gemma 3 Specifics !!! ---
#         try:
#             if GEMMA_LIB_AVAILABLE and hasattr(gemma, 'Gemma') and hasattr(gemma.Gemma, 'from_preset'):
#                  logger.info("Attempting load using gemma.Gemma.from_preset...")
#                  # This function should handle finding tokenizer.model within the preset dir
#                  model = gemma.Gemma.from_preset(self.preset_path)
#                  logger.info("Model loaded successfully using gemma.Gemma.from_preset.")
#                  # Verify tokenizer was loaded
#                  if not hasattr(model, 'tokenizer') or not model.tokenizer:
#                      logger.warning("gemma.Gemma.from_preset loaded model but tokenizer is missing!")
#                  return model
#             else:
#                  logger.warning("Official 'gemma' library/method not available.")
#                  raise NotImplementedError("Gemma library unavailable or from_preset missing.")

#             # Add other loading options (Keras, TF Hub) here if needed,
#             # but ensure they also somehow provide access to the correct tokenizer.

#         except NotImplementedError:
#              logger.warning("No suitable loading method found. Using a dummy placeholder.")
#              # Fallback to a placeholder - REMOVE THIS IN REAL IMPLEMENTATION
#              # ... (Dummy Keras model definition - same as before, but won't have .tokenizer) ...
#              input_text = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_tokens")
#              input_image = keras.layers.Input(shape=(896, 896, 3), dtype=tf.bfloat16, name="input_image") # Use known size
#              text_emb = keras.layers.Embedding(262144, 1024)(input_text) # Match tokenizer size approx
#              img_emb = keras.layers.Conv2D(1024, (16, 16), strides=(16,16))(input_image) # Dummy patch embedding
#              img_emb = keras.layers.Reshape((-1, 1024))(img_emb) # Flatten patches
#              # Dummy multimodal combination - extremely simplified
#              combined = keras.layers.Concatenate(axis=1)([keras.layers.GlobalAveragePooling1D()(text_emb)[tf.newaxis, :], keras.layers.GlobalAveragePooling1D()(img_emb)])
#              output = keras.layers.Dense(262144, name="dummy_output_logits")(combined) # Dummy output logits
#              model = keras.Model(inputs=[input_text, input_image], outputs=output, name="DummyGemmaPlaceholder")
#              logger.warning("Initialized a DUMMY model placeholder. Replace _load_pretrained_model_with_tokenizer with actual loading code.")
#              return model # This dummy model won't have a .tokenizer attribute

#         except Exception as e:
#              logger.error(f"Failed to load Gemma 3 model from {self.preset_path}: {e}", exc_info=True)
#              return None
#         # --- !!! END Placeholder !!! ---

#     def call(self, inputs, training: bool = False):
#         """Defines the forward pass for the model."""
#         # ... (Keep placeholder implementation as before, needs specifics) ...
#         if not self.gemma_model:
#             logger.error("Gemma model not loaded, cannot execute call.")
#             return None
#         try:
#             logger.warning("GemmaVisionModel.call() is a placeholder, passing input to underlying model without verification.")
#             return self.gemma_model(inputs, training=training)
#         except Exception as e:
#              logger.error(f"Error during GemmaVisionModel.call: {e}", exc_info=True)
#              return None


#     # def generate(self,
#     #              prompts: Union[List[str], str],
#     #              images: Optional[Union[tf.Tensor, List[Optional[tf.Tensor]]]] = None,
#     #              max_length: int = 1024,
#     #              temperature: float = 0.7,
#     #              top_k: int = 40
#     #              ) -> Union[List[str], str]:
#     #     """Generates text based on text prompt(s) and optional image(s)."""
#     #     # Use the tokenizer loaded by the model
#     #     if not self.gemma_model or not self.tokenizer:
#     #         logger.error("Model or its tokenizer not loaded correctly, cannot generate.")
#     #         return ["Error: Model/Tokenizer not loaded."] * (len(prompts) if isinstance(prompts, list) else 1)

#     #     is_batch = isinstance(prompts, list)
#     #     if not is_batch:
#     #         prompts = [prompts]
#     #         images = [images]

#     #     if images and len(prompts) != len(images):
#     #          logger.error(f"Number of prompts ({len(prompts)}) does not match number of images ({len(images)}).")
#     #          return ["Error: Prompt/Image mismatch."] * len(prompts)

#     #     # --- 1. Prepare Formatted Prompts ---
#     #     # Use constants defined earlier (either from gemma lib or fallbacks)
#     #     formatted_prompts = []
#     #     for prompt_content in prompts:
#     #          prompt_content = str(prompt_content).strip()
#     #          formatted = f"{BOS_TOKEN}{START_OF_TURN}{USER}\n{prompt_content}{END_OF_TURN}{START_OF_TURN}{MODEL}\n"
#     #          formatted_prompts.append(formatted)

#     #     # --- 2. Tokenize ---
#     #     try:
#     #         # Use the model's tokenizer
#     #         # Assuming tokenizer.encode handles list input for batching if generate needs lists
#     #         # Or maybe the generate method itself takes the raw formatted strings? Check docs.
#     #         # Let's assume we tokenize first.
#     #         tokenized_prompts = [self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in formatted_prompts]
#     #         # Need to prepare inputs correctly for the underlying generate method
#     #         # processed_inputs = tokenized_prompts # Or maybe padded tensors?
#     #         processed_inputs = formatted_prompts # Many high-level generate methods take strings

#     #     except Exception as e:
#     #          logger.error(f"Error during prompt tokenization: {e}", exc_info=True)
#     #          return [f"Error tokenizing prompt."] * len(prompts)

#     #     # --- 3. Prepare Image Inputs ---
#     #     # ... (Keep placeholder image processing logic as before) ...
#     #     processed_images = None
#     #     if images:
#     #          try:
#     #               valid_images = [img for img in images if img is not None]
#     #               if valid_images:
#     #                   processed_images = tf.stack(valid_images)
#     #          except Exception as e:
#     #               logger.error(f"Error processing image batch: {e}", exc_info=True)
#     #               return [f"Error processing images."] * len(prompts)

#     #     # --- 4. Generate Tokens ---
#     #     logger.info(f"Starting generation for {len(prompts)} prompt(s)...")
#     #     try:
#     #         # --- !!! Placeholder for actual generation call !!! ---
#     #         # This call needs to match the API of the loaded self.gemma_model
#     #         # It might take strings, lists of tokens, image tensors/embeddings etc.

#     #         # Example if using gemma library's generate
#     #         # sampler_obj = sampler.TopKSampler(k=top_k, temperature=temperature) # Example
#     #         # result_sequences = self.gemma_model.generate(
#     #         #     processed_inputs, # Likely strings or token lists
#     #         #     images=processed_images, # Hypothetical
#     #         #     max_length=max_length,
#     #         #     sampler=sampler_obj
#     #         # ) # Returns list of token lists or strings? Check docs

#     #         # --- Using a dummy response ---
#     #         logger.warning("Using DUMMY generation response. Replace with actual model.generate() call.")
#     #         # Simulate just returning the prompt + EOS id
#     #         result_sequences = [[*p_tokens, self.eos_token_id] for p_tokens in tokenized_prompts]


#     #     except Exception as e:
#     #          logger.error(f"Error during model generation: {e}", exc_info=True)
#     #          return [f"Error during generation."] * len(prompts)

#     #     # --- 5. Decode Output ---
#     #     generated_texts = []
#     #     try:
#     #         # Assuming result_sequences is a list of lists of token IDs
#     #         for i, sequence in enumerate(result_sequences):
#     #              prompt_len = len(tokenized_prompts[i])
#     #              if len(sequence) > prompt_len:
#     #                   new_tokens = sequence[prompt_len:]
#     #              else:
#     #                   new_tokens = [] # Assume nothing new generated if sequence not longer

#     #              if hasattr(new_tokens, 'numpy'): new_tokens = new_tokens.numpy()
#     #              if not isinstance(new_tokens, list): new_tokens = new_tokens.tolist()

#     #              if new_tokens and new_tokens[-1] == self.eos_token_id:
#     #                   new_tokens = new_tokens[:-1]

#     #              decoded_text = self.tokenizer.decode(new_tokens)
#     #              generated_texts.append(decoded_text.strip())

#     #         logger.info(f"Generation completed for {len(prompts)} prompt(s).")

#     #     except Exception as e:
#     #         logger.error(f"Error during token decoding: {e}", exc_info=True)
#     #         while len(generated_texts) < len(prompts):
#     #              generated_texts.append("Error decoding output.")

#     #     return generated_texts[0] if not is_batch else generated_texts

#     def generate(self,
#                     prompts: Union[List[str], str],
#                     images: Optional[Union[tf.Tensor, List[Optional[tf.Tensor]]]] = None,
#                     max_length: int = 1024,
#                     temperature: float = 0.7, # Example sampling param
#                     top_k: int = 40         # Example sampling param
#                     ) -> Union[List[str], str]:
#             """Generates text based on text prompt(s) and optional image(s)."""
#             # Use the tokenizer loaded by the model
#             if not self.gemma_model or not self.tokenizer:
#                 logger.error("Model or its tokenizer not loaded correctly, cannot generate.")
#                 return ["Error: Model/Tokenizer not loaded."] * (len(prompts) if isinstance(prompts, list) else 1)

#             is_batch = isinstance(prompts, list)
#             if not is_batch:
#                 prompts = [prompts]
#                 images = [images] # Wrap single image/None in a list

#             if images and len(prompts) != len(images):
#                 logger.error(f"Number of prompts ({len(prompts)}) does not match number of images ({len(images)}).")
#                 return ["Error: Prompt/Image mismatch."] * len(prompts)

#             # --- 1. Prepare Formatted Prompts for Model ---
#             formatted_prompts = []
#             for prompt_content in prompts:
#                 prompt_content = str(prompt_content).strip()
#                 formatted = f"{BOS_TOKEN}{START_OF_TURN}{USER}\n{prompt_content}{END_OF_TURN}{START_OF_TURN}{MODEL}\n"
#                 formatted_prompts.append(formatted)

#             # --- 2. Tokenize (Potentially needed for decoding later or if model takes tokens) ---
#             try:
#                 # We still need the tokenized version to calculate prompt length for decoding
#                 tokenized_prompts = [self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in formatted_prompts]
#             except Exception as e:
#                 logger.error(f"Error during prompt tokenization: {e}", exc_info=True)
#                 return [f"Error tokenizing prompt."] * len(prompts)

#             # --- 3. Prepare Image Inputs (if any) ---
#             processed_images = None
#             if images:
#                 try:
#                     valid_images = [img for img in images if img is not None]
#                     if valid_images:
#                         processed_images = tf.stack(valid_images) # Assumes all are tensors of same shape
#                 except Exception as e:
#                     logger.error(f"Error processing image batch: {e}", exc_info=True)
#                     return [f"Error processing images."] * len(prompts)

#             # --- 4. Generate Tokens ---
#             logger.info(f"Starting generation for {len(prompts)} prompt(s)...")
#             try:
#                 # --- !!! CORRECTED: Call the underlying model's generate method !!! ---
#                 # Ensure arguments match how the mock expects to be called, OR
#                 # how the real Gemma 3 generate method works.
#                 # Passing formatted_prompts (list of strings) here, as the mock handles it.
#                 result_sequences = self.gemma_model.generate(
#                     formatted_prompts, # Pass formatted strings
#                     images=processed_images,
#                     max_length=max_length,
#                     # Add other necessary args like temperature, top_k if generate accepts them directly
#                     # temperature=temperature,
#                     # top_k=top_k
#                 )
#                 # --- !!! END CORRECTION !!! ---

#             except Exception as e:
#                 logger.error(f"Error during model generation: {e}", exc_info=True)
#                 return [f"Error during generation."] * len(prompts)

#             # --- 5. Decode Output ---
#             # (Decoding logic remains the same)
#             generated_texts = []
#             try:
#                 # Assuming result_sequences is a list of lists of token IDs (mock returns this)
#                 for i, sequence in enumerate(result_sequences):
#                     prompt_len = len(tokenized_prompts[i])
#                     if len(sequence) > prompt_len:
#                         new_tokens = sequence[prompt_len:]
#                     else:
#                         new_tokens = []

#                     if hasattr(new_tokens, 'numpy'): new_tokens = new_tokens.numpy()
#                     if not isinstance(new_tokens, list): new_tokens = new_tokens.tolist()

#                     # Use self.eos_token_id if available, otherwise fallback
#                     eos_id_to_check = getattr(self, 'eos_token_id', 1) # Default to 1 if not set
#                     if new_tokens and new_tokens[-1] == eos_id_to_check:
#                         new_tokens = new_tokens[:-1]

#                     decoded_text = self.tokenizer.decode(new_tokens)
#                     generated_texts.append(decoded_text.strip())

#                 logger.info(f"Generation completed for {len(prompts)} prompt(s).")

#             except Exception as e:
#                 logger.error(f"Error during token decoding: {e}", exc_info=True)
#                 while len(generated_texts) < len(prompts):
#                     generated_texts.append("Error decoding output.")

#             return generated_texts[0] if not is_batch else generated_texts

# # --- Example Usage (for direct testing) ---
# if __name__ == '__main__':
#     print("Running GemmaVisionModel example usage...")
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     current_dir = Path(__file__).parent
#     project_root_path = current_dir.parent.parent
#     cfg_path = project_root_path / "configs/gemma3_4B_config.yaml"

#     print(f"Project Root: {project_root_path}")
#     print(f"Config Path: {cfg_path}")

#     if not cfg_path.exists():
#         print(f"ERROR: Config file not found at {cfg_path}")

#     # --- Load the model (no tokenizer path needed) ---
#     model_wrapper = GemmaVisionModel(config_path=str(cfg_path))

#     # --- Test generation (only if model and its tokenizer loaded) ---
#     if model_wrapper.gemma_model and model_wrapper.tokenizer:
#         test_image_tensor = None # Placeholder
#         prompt1 = "Explain the concept of KL divergence simply."
#         prompt2 = f"{IMAGE_TOKEN}\nWhat objects are in this image?" if test_image_tensor is not None else "What is the capital of France?"
#         prompts_to_test = [prompt1, prompt2]
#         images_to_test = [None, test_image_tensor]

#         print(f"\n--- Running Generation Test ---")
#         outputs = model_wrapper.generate(
#             prompts=prompts_to_test,
#             images=images_to_test,
#             max_length=64
#         )

#         print("\n--- Generated Outputs ---")
#         for i, output in enumerate(outputs):
#             print(f"Prompt {i+1}: {prompts_to_test[i]}")
#             print(f"Output {i+1}: {output}\n")
#     else:
#         print("\nSkipping generation test as model or tokenizer failed to load correctly.")



# src/model/gemma_vision.py

import os
import yaml
import logging
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf # Keep for potential input type check / conversion
from typing import Optional, List, Dict, Union, Any

# Attempt to import the official Gemma library and components
try:
    import gemma as gm # Use alias gm
    # Explicitly import components we expect to use based on API list
    from gemma import sampler as gm_sampler
    from gemma import checkpoints as gm_ckpts
    from gemma import layers as gm_nn # nn seems to be under layers based on API list? Double check. If not use gm.nn directly if top-level
    from gemma import text as gm_text
    GEMMA_LIB_AVAILABLE = True
    # Define constants based on library availability or defaults
    # These might be better accessed via tokenizer instance once loaded
    IMAGE_TOKEN = '<img>' # Default placeholder
    START_OF_TURN = '<start_of_turn>'
    END_OF_TURN = '<end_of_turn>'
    BOS_TOKEN = '<bos>'
    USER = 'user'
    MODEL = 'model'

except ImportError:
    logging.warning("Could not import official 'gemma' library or submodules. Using placeholders/defaults. Cannot load real model.")
    GEMMA_LIB_AVAILABLE = False
    # Define dummy classes/functions if library is missing for basic structure
    gm = None
    gm_sampler = MagicMock() # Use MagicMock if testing framework available, else None/dummy class
    gm_ckpts = MagicMock()
    gm_nn = MagicMock()
    gm_text = MagicMock()
    # Define fallback constants
    IMAGE_TOKEN = '<img>'
    START_OF_TURN = '<start_of_turn>'
    END_OF_TURN = '<end_of_turn>'
    BOS_TOKEN = '<bos>'
    USER = 'user'
    MODEL = 'model'
    # Add dummy classes for type hints if needed and lib missing
    if 'MagicMock' not in globals(): # Avoid redefining if imported from test later
        class MagicMock: pass # Simplest dummy


# Setup logger for this module
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    config_path_obj = Path(config_path)
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

class GemmaVisionModel: # Plain Python class
    """
    Wrapper class for loading and interacting with a pre-trained Gemma 3 multimodal model
    using the official JAX-based 'gemma' library.
    """
    def __init__(self, config_path: str):
        """Initializes the model wrapper."""
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        self._set_paths()

        # --- Load Components ---
        self.tokenizer = self._load_tokenizer()
        self.gemma_model_arch = self._load_architecture()
        self.gemma_model_params = self._load_parameters()

        # --- Store Token IDs ---
        self._set_token_ids()

        # Log status
        if not all([self.tokenizer, self.gemma_model_arch, self.gemma_model_params]):
            logger.error("Model initialization failed due to missing components.")
        else:
            logger.info("GemmaVisionModel initialized successfully.")

    def _set_paths(self):
        """Resolves paths from config or defaults."""
        default_preset_dir = "pre-trained/gemma3-4b"
        preset_relative_path = self.config.get("model", {}).get("preset_path", default_preset_dir)
        self.preset_path = str(self.project_root / preset_relative_path)
        logger.info(f"Resolved model preset path to: {self.preset_path}")

    def _load_tokenizer(self) -> Optional[Any]:
        """Loads the Gemma3Tokenizer."""
        if not gm_text or not hasattr(gm_text, 'Gemma3Tokenizer'):
            logger.error("gemma.text.Gemma3Tokenizer not available.")
            return None
        tokenizer_model_file = Path(self.preset_path) / 'tokenizer.model'
        if not tokenizer_model_file.is_file():
            logger.error(f"tokenizer.model not found inside preset path: {self.preset_path}")
            return None
        try:
            # Pass the directory containing tokenizer.model
            tokenizer = gm_text.Gemma3Tokenizer(path=self.preset_path)
            logger.info(f"Tokenizer loaded via gm.text.Gemma3Tokenizer from {self.preset_path} (Vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}).")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {self.preset_path}: {e}", exc_info=True)
            return None

    def _load_architecture(self) -> Optional[Any]:
        """Instantiates the Gemma model architecture."""
        # Assuming gm.nn was imported correctly or gm_nn points to it
        if not gm_nn or not hasattr(gm_nn, 'Gemma3_4B'):
            logger.error("gemma.nn.Gemma3_4B not available. Cannot instantiate architecture.")
            return None
        try:
            logger.info("Instantiating Gemma 3 4B architecture...")
            # Check if constructor needs arguments based on API (example didn't show required args)
            arch = gm_nn.Gemma3_4B()
            logger.info("Gemma 3 4B architecture instantiated.")
            return arch
        except Exception as e:
            logger.error(f"Failed to instantiate Gemma 3 4B architecture: {e}", exc_info=True)
            return None

    def _load_parameters(self) -> Optional[Any]:
        """Loads the pre-trained model parameters."""
        if not gm_ckpts or not hasattr(gm_ckpts, 'load_params'):
            logger.error("gemma.ckpts.load_params not available. Cannot load parameters.")
            return None
        preset_path_obj = Path(self.preset_path)
        if not preset_path_obj.exists() or not preset_path_obj.is_dir():
             logger.error(f"Model preset directory not found: {self.preset_path}")
             return None
        try:
            logger.info(f"Loading model parameters using gm.ckpts.load_params from {self.preset_path}...")
            params = gm_ckpts.load_params(checkpoint_path=self.preset_path)
            logger.info(f"Model parameters loaded successfully (Type: {type(params)}).")
            return params
        except Exception as e:
            logger.error(f"Failed to load model parameters from {self.preset_path}: {e}", exc_info=True)
            return None

    def _set_token_ids(self):
        """Sets special token IDs using the loaded tokenizer."""
        self.bos_token_id = getattr(self.tokenizer, 'bos_id', 2) # Default 2
        self.eos_token_id = getattr(self.tokenizer, 'eos_id', 1) # Default 1
        self.pad_token_id = getattr(self.tokenizer, 'pad_id', 0) # Default 0
        # Get string representations if available, otherwise use defaults
        global BOS_TOKEN, IMAGE_TOKEN, START_OF_TURN, END_OF_TURN, USER, MODEL # Allow modification
        BOS_TOKEN = getattr(self.tokenizer, 'bos_token', BOS_TOKEN)
        # IMAGE_TOKEN likely not part of standard tokenizer, keep default
        # START_OF_TURN etc. likely not direct attributes, keep defaults

    def _preprocess_image(self, image: Union[tf.Tensor, np.ndarray]) -> Optional[jnp.ndarray]:
        """Converts/preprocesses image for JAX Gemma model."""
        # --- Placeholder Preprocessing - NEEDS VERIFICATION from Gemma 3 Docs ---
        try:
            target_size = 896 # Common size for vision models
            target_shape = (target_size, target_size)

            if isinstance(image, tf.Tensor): img_np = image.numpy()
            elif isinstance(image, np.ndarray): img_np = image
            else: raise TypeError(f"Unsupported image type: {type(image)}")

            # Ensure correct channel number (e.g., handle grayscale)
            if img_np.ndim == 2: # Grayscale
                 img_np = np.stack((img_np,) * 3, axis=-1)
            elif img_np.ndim == 3 and img_np.shape[-1] == 1: # Grayscale with channel dim
                 img_np = np.concatenate((img_np,) * 3, axis=-1)
            elif img_np.ndim != 3 or img_np.shape[-1] != 3:
                 raise ValueError(f"Unsupported image shape: {img_np.shape}")

            # Resize (Using JAX - might need external lib like skimage or cv2 if JAX doesn't have robust resize)
            # jax.image.resize is available
            # Ensure input is float for resize
            img_np_float = img_np.astype(np.float32)
            img_resized_jax = jax.image.resize(
                img_np_float,
                shape=(target_size, target_size, 3), # Target shape HWC
                method='bilinear' # Or 'lanczos3', 'lanczos5'
            )

            # Add "sequence" dimension? -> (1, H, W, C) ?? Check model spec
            img_rearranged = img_resized_jax[jnp.newaxis, ...]

            # Cast to expected dtype (bfloat16 is common)
            img_final = img_rearranged.astype(jnp.bfloat16)

            logger.debug(f"Image preprocessed to shape: {img_final.shape}, dtype: {img_final.dtype}")
            return img_final

        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}", exc_info=True)
            return None
        # --- End Placeholder Preprocessing ---

    def generate(self,
                 prompts: Union[List[str], str],
                 images: Optional[Union[tf.Tensor, np.ndarray, List[Optional[Union[tf.Tensor, np.ndarray]]]]] = None,
                 max_length: int = 1024,
                 temperature: float = 0.7,
                 top_k: Optional[int] = None, # Make K and P optional
                 top_p: Optional[float] = None
                 ) -> Union[List[str], str]:
        """Generates text based on text prompt(s) and optional image(s) using JAX/Gemma."""
        if not self.gemma_model_arch or not self.gemma_model_params or not self.tokenizer:
            logger.error("Model architecture, parameters, or tokenizer not loaded correctly, cannot generate.")
            error_msg = "Error: Model/Params/Tokenizer not loaded."
            return [error_msg] * (len(prompts) if isinstance(prompts, list) else 1)

        is_batch = isinstance(prompts, list)
        if not is_batch:
            prompts = [prompts]
            images = [images]

        if images and len(prompts) != len(images):
             logger.error(f"Batch size mismatch: {len(prompts)} prompts vs {len(images)} images.")
             return ["Error: Prompt/Image mismatch."] * len(prompts)

        # --- 1. Prepare Inputs ---
        input_token_ids_list = []
        processed_images_list = []
        max_prompt_len = 0

        for i, prompt_content in enumerate(prompts):
            prompt_content = str(prompt_content).strip()
            # Add IMAGE_TOKEN if image present for this item? Assumes it's already in prompt_content.
            # Let's assume prompt_content received here already contains <img> if needed.
            formatted_prompt = f"{BOS_TOKEN}{START_OF_TURN}{USER}\n{prompt_content}{END_OF_TURN}{START_OF_TURN}{MODEL}\n"
            try:
                 tokens = self.tokenizer.encode(formatted_prompt, add_bos=False, add_eos=False)
                 input_token_ids_list.append(tokens)
                 max_prompt_len = max(max_prompt_len, len(tokens))
            except Exception as e:
                 logger.error(f"Error tokenizing prompt {i}: {e}", exc_info=True)
                 return [f"Error tokenizing prompt {i}."] * len(prompts)

            processed_img_jnp = None
            if images and images[i] is not None:
                 processed_img_jnp = self._preprocess_image(images[i])
                 if processed_img_jnp is None:
                      logger.error(f"Image processing failed for prompt {i}.")
                      return [f"Error processing image {i}."] * len(prompts)
            processed_images_list.append(processed_img_jnp) # Contains None or JAX array


        # --- 2. Pad token inputs & Prepare Batch ---
        pad_value = self.pad_token_id
        padded_token_ids = np.full((len(input_token_ids_list), max_prompt_len), pad_value, dtype=np.int32)
        for i, tokens in enumerate(input_token_ids_list):
             padded_token_ids[i, :len(tokens)] = tokens
        input_token_ids_jax = jnp.asarray(padded_token_ids)

        # --- 3. Prepare Image Batch ---
        # How to handle None images in a batch for JAX model? Highly dependent on model arch.
        # Simplification: If *any* image is present, create a stacked batch. Assume model handles it.
        # If no images present, pass None. If mixing is needed, model API must support it (or use loops).
        final_image_input = None
        valid_images_jnp = [img for img in processed_images_list if img is not None]
        if valid_images_jnp:
             if len(valid_images_jnp) != len(prompts):
                 # This case requires the model to handle missing images, e.g., via masking or flags.
                 # For now, we'll log a warning and proceed assuming the model might handle `None` implicitly,
                 # or raise an error if stacking fails.
                 logger.warning("Batch contains a mix of image and text-only prompts. Model must support this.")
                 # Attempting to stack might fail if shapes mismatch due to None -> How JAX handles this?
                 # Let's pass the list of images (with Nones) if stacking fails.
                 # This is highly speculative.
                 final_image_input = processed_images_list # Pass list hoping model handles it
             else:
                 try:
                     final_image_input = jnp.concatenate(valid_images_jnp, axis=0) # Concatenate along batch dim
                 except Exception as e:
                     logger.error(f"Could not stack image tensors for batch: {e}. Passing list.")
                     final_image_input = processed_images_list # Fallback to list

        # --- 4. Select Sampler ---
        if top_p is not None and top_p > 0.0:
            sampler_obj = gm_sampler.TopPSampler(p=top_p, temperature=temperature)
            logger.info(f"Using Top-P sampler (p={top_p}, temp={temperature})")
        elif top_k is not None and top_k > 0:
            sampler_obj = gm_sampler.TopkSampling(k=top_k, temperature=temperature) # Corrected class name? API List had TopkSampling
            logger.info(f"Using Top-K sampler (k={top_k}, temp={temperature})")
        else:
            sampler_obj = gm_sampler.Greedy() # Use Greedy sampler if neither top_k nor top_p
            logger.info("Using Greedy sampler")

        # --- 5. Generate Tokens ---
        logger.info(f"Starting generation for batch size {len(prompts)}...")
        output_token_ids = None
        try:
            # --- !!! Placeholder: Actual JAX/Gemma generation call !!! ---
            # This needs the exact function/method and signature from the library.
            # Common pattern: model_state = model.init(...) ; output = model.apply(model_state, ...)
            # Or the loaded params might be sufficient: model_arch.apply({'params': params}, ...)
            # Generation might be a specific method or part of apply.

            variables = {'params': self.gemma_model_params}
            # Model might expect inputs in a specific dictionary structure
            model_inputs = {'input_tokens': input_token_ids_jax} # Use a key model expects
            if final_image_input is not None:
                 # Key name 'images' or 'image_features' etc. depends on gm.nn.Gemma3_4B
                 model_inputs['images'] = final_image_input

            # Calculate number of steps to generate
            max_decode_steps = max_length - input_token_ids_jax.shape[-1]
            if max_decode_steps <= 0:
                 logger.warning("max_length <= prompt length. Returning empty response.")
                 return [""] * len(prompts)

            # Create a dummy RNG key - real applications need proper key management
            rng_key = jax.random.PRNGKey(0)

            # ASSUMPTION: The architecture object has a `generate` method usable with apply
            # OR maybe a top-level gemma.generate function exists?
            # Trying apply with a 'generate' method name is a guess.
            logger.info(f"Calling model apply/generate (placeholder) with input shapes: "
                        f"tokens={input_token_ids_jax.shape}, "
                        f"images={final_image_input.shape if final_image_input is not None else 'None'}")

            # output_token_ids = self.gemma_model_arch.apply(
            #     variables,
            #     inputs=model_inputs, # Pass dict directly? Or specific args?
            #     method=self.gemma_model_arch.generate, # Is 'generate' the method name?
            #     sampler=sampler_obj,
            #     rngs={'sampler': rng_key}, # Pass RNG if sampler needs it
            #     total_generation_steps=max_decode_steps
            # )
            # If no obvious generate method, maybe __call__ handles it?
            # output_token_ids = self.gemma_model_arch.apply(
            #     variables,
            #     model_inputs, # Pass dict?
            #     sampler=sampler_obj, # How sampler passed?
            #     # ... other args ...
            # )

            # --- Using Dummy Output ---
            logger.warning("Using DUMMY generation output tokens. Replace with actual model generation call.")
            output_shape = (input_token_ids_jax.shape[0], max_decode_steps)
            # Generate some dummy tokens besides EOS for testing decode
            dummy_tokens = jnp.array([[ord('O'), ord('u'), ord('t')]* (max_decode_steps//3 +1)], dtype=jnp.int32)
            output_token_ids = jnp.tile(dummy_tokens, (output_shape[0], 1))[:, :max_decode_steps]
            # --- End Dummy Output ---

            if output_token_ids is None:
                 raise ValueError("Model generation call returned None.")
            logger.info(f"Model generation call completed (placeholder logic). Output shape: {output_token_ids.shape}")

        except Exception as e:
             logger.error(f"Error during model generation call: {e}", exc_info=True)
             return [f"Error during generation."] * len(prompts)

        # --- 6. Decode Output ---
        generated_texts = []
        logger.info("Decoding generated tokens...")
        try:
            output_token_ids_np = np.array(output_token_ids)
            for i, sequence_ids in enumerate(output_token_ids_np):
                 token_list = sequence_ids.tolist()
                 eos_id_to_check = self.eos_token_id
                 pad_id_to_check = self.pad_token_id

                 # Find first EOS or PAD
                 end_indices = []
                 try: end_indices.append(token_list.index(eos_id_to_check))
                 except ValueError: pass
                 try: end_indices.append(token_list.index(pad_id_to_check))
                 except ValueError: pass

                 end_index = min(end_indices) if end_indices else len(token_list)
                 final_tokens = token_list[:end_index]

                 decoded_text = self.tokenizer.decode(final_tokens)
                 generated_texts.append(decoded_text.strip())
            logger.info(f"Decoding completed for {len(prompts)} sequence(s).")
        except Exception as e:
            logger.error(f"Error during token decoding: {e}", exc_info=True)
            while len(generated_texts) < len(prompts): generated_texts.append("Error decoding output.")

        return generated_texts[0] if not is_batch else generated_texts

# --- Example Usage (for direct testing) ---
if __name__ == '__main__':
    # Setup basic logging for the test run
    # Basic config should be sufficient if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger.setLevel(logging.INFO)
    print(f"Running GemmaVisionModel example usage (JAX version) from {__file__}...")

    current_dir = Path(__file__).parent
    project_root_path = current_dir.parent.parent
    cfg_path = project_root_path / "configs/gemma3_4B_config.yaml"

    print(f"Project Root: {project_root_path}")
    print(f"Config Path: {cfg_path}")

    if not cfg_path.exists():
        print(f"ERROR: Config file not found at {cfg_path}")
        # Attempt to proceed without config for placeholder testing
        # sys.exit(1) # Don't exit if only testing placeholders

    # --- Load the model ---
    model_wrapper = GemmaVisionModel(config_path=str(cfg_path))

    # --- Test generation (only if components loaded) ---
    if model_wrapper.gemma_model_arch and model_wrapper.gemma_model_params and model_wrapper.tokenizer:
        logger.info("Model components appear loaded (or placeholders active). Attempting generation.")
        # Use numpy for dummy image creation
        dummy_image_np = np.random.randint(0, 256, size=(896, 896, 3), dtype=np.uint8)

        prompt1 = "Describe the concept of KL divergence in simple terms."
        prompt2 = f"{IMAGE_TOKEN}\nDescribe this image in detail."
        prompt3 = "What is the capital of France?"
        prompts_to_test = [prompt1, prompt2, prompt3]
        images_to_test = [None, dummy_image_np, None] # Pass np array

        print(f"\n--- Running Generation Test (Batch Size: {len(prompts_to_test)}) ---")
        outputs = model_wrapper.generate(
            prompts=prompts_to_test,
            images=images_to_test,
            max_length=64,
            temperature=0.1, # Low temp for dummy test
            top_k=1          # Greedy for dummy test
        )

        print("\n--- Generated Outputs ---")
        if isinstance(outputs, list):
            for i, output in enumerate(outputs):
                print(f"Prompt {i+1}: {prompts_to_test[i]}")
                print(f"Output {i+1}: {output}\n")
        else:
             print(f"Output: {outputs}")
    else:
        print("\nSkipping generation test as model architecture, parameters, or tokenizer failed to load correctly during __init__.")