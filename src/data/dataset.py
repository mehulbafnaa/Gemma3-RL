# """
# Loads and prepares the MathVista dataset for use with Gemma 3,
# creating a TensorFlow Dataset pipeline.
# """

# import os # <-- Added import for os.path.basename
# import sys
# import logging
# import zipfile
# import io
# from pathlib import Path
# from typing import Dict, List, Optional, Any
# import numpy as np
# import requests
# from tqdm import tqdm
# from PIL import Image
# import sentencepiece as spm
# import tensorflow as tf
# # --- Corrected Imports ---
# from datasets import load_dataset
# from huggingface_hub.errors import HfHubHTTPError # Import the correct error
# # --- End Corrected Imports ---

# # Assuming preprocessing functions are in the same directory or package
# from .preprocessing import format_gemma_prompt, resize_image_for_gemma

# logger = logging.getLogger(__name__)


# # Updated download function
# def download_mathvista_images(data_dir: str) -> str:
#     """
#     Downloads and extracts MathVista images, trying to find the correct image folder.
#     """
#     images_dir_base = Path(data_dir) / "images" # The target directory name
#     # Check if directory exists and is not empty (contains image files)
#     if images_dir_base.exists() and any(f.suffix.lower() in ['.jpg', '.png', '.jpeg'] for f in images_dir_base.glob('*') if f.is_file()):
#         logger.info(f"Images already found directly in {images_dir_base}")
#         return str(images_dir_base)

#     # Check if a subdirectory *within* images_dir_base contains images (common zip structure)
#     if images_dir_base.exists():
#         subdirs = [d for d in images_dir_base.iterdir() if d.is_dir()]
#         for subdir in subdirs:
#              if any(f.suffix.lower() in ['.jpg', '.png', '.jpeg'] for f in subdir.glob('*') if f.is_file()):
#                   logger.info(f"Images found in subdirectory: {subdir}. Assuming this is the correct path.")
#                   return str(subdir) # Return the path to the subdirectory containing images

#     # If images not found, proceed with download
#     images_dir_base.mkdir(parents=True, exist_ok=True)
#     url = "https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/images.zip"
#     logger.info(f"Downloading MathVista images from {url}...")

#     try:
#         response = requests.get(url, stream=True, timeout=300) # Add timeout
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

#         total_size = int(response.headers.get('content-length', 0))
#         block_size = 8192 # Use larger block size for faster download
#         progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading images", leave=False)

#         zip_content = io.BytesIO()
#         for data in response.iter_content(block_size):
#             progress_bar.update(len(data))
#             zip_content.write(data)
#         progress_bar.close()

#         if total_size != 0 and progress_bar.n != total_size:
#             logger.warning("Downloaded size does not match expected size.")
#             # Decide whether to proceed or raise error

#         logger.info("Download complete. Extracting images...")
#         zip_content.seek(0) # Reset buffer position
#         with zipfile.ZipFile(zip_content) as z:
#             z.extractall(data_dir) # Extract contents into data_dir

#         # --- Post-Extraction Verification ---
#         final_images_dir = images_dir_base # Assume default first

#         # Check 1: Are images directly in data_dir/images?
#         if images_dir_base.exists() and any(f.suffix.lower() in ['.jpg', '.png', '.jpeg'] for f in images_dir_base.glob('*') if f.is_file()):
#              final_images_dir = images_dir_base
#              logger.info(f"Verified images directly in: {final_images_dir}")

#         # Check 2: If not found directly, is there ONE subdirectory inside data_dir/images that contains images?
#         #          This handles zips that contain a single top-level folder (e.g., images.zip -> extracts 'images/' folder)
#         else:
#             subdirs_with_images = []
#             if images_dir_base.exists():
#                 for item in images_dir_base.iterdir():
#                     if item.is_dir():
#                          if any(f.suffix.lower() in ['.jpg', '.png', '.jpeg'] for f in item.glob('*') if f.is_file()):
#                              subdirs_with_images.append(item)

#             if len(subdirs_with_images) == 1:
#                 final_images_dir = subdirs_with_images[0]
#                 logger.warning(f"Images not found directly in {images_dir_base}, but found in single subdirectory: {final_images_dir}. Using this path.")
#             elif len(subdirs_with_images) > 1:
#                  logger.error(f"Found multiple subdirectories containing images within {images_dir_base}: {subdirs_with_images}. Ambiguous structure.")
#                  raise FileNotFoundError(f"Ambiguous image directory structure in {images_dir_base}")
#             else:
#                  logger.error(f"Could not find image files directly in {images_dir_base} or within a single clear subdirectory after extraction.")
#                  raise FileNotFoundError(f"Could not locate extracted image files structure in {images_dir_base}")

#         logger.info(f"MathVista images available at {final_images_dir}")
#         return str(final_images_dir) # Return the determined correct path

#     except requests.exceptions.RequestException as e:
#         logger.error(f"Failed to download images from {url}: {e}")
#         raise
#     except zipfile.BadZipFile:
#         logger.error(f"Downloaded file from {url} is not a valid zip file.")
#         raise
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during image download/extraction: {e}")
#         raise


# class MathVistaDataset:
#     """
#     Loads and preprocesses the MathVista dataset for Gemma 3 into a TensorFlow pipeline.

#     Attributes:
#         data_dir (str): Root data directory.
#         images_dir (str): Directory containing MathVista images.
#         split (str): Dataset split to load (e.g., 'testmini', 'train').
#         batch_size (int): Batch size for the TensorFlow dataset.
#         image_size (int): Target size for image resizing (square).
#         max_length (int): Maximum sequence length for tokenization (if used).
#         tokenizer (Optional[spm.SentencePieceProcessor]): Loaded tokenizer.
#         examples (List[Dict]): List of prepared example dictionaries (metadata).
#         tf_dataset (tf.data.Dataset): The final TensorFlow dataset pipeline.
#     """

#     def __init__(
#         self,
#         data_dir: str,
#         images_dir: str, # Require images_dir to be explicitly provided
#         split: str = "testmini",
#         batch_size: int = 32,
#         tokenizer_path: Optional[str] = None,
#         image_size: int = 896,
#         max_length: int = 2048,
#         hf_token: Optional[str] = None, # Optional HF token for private/gated datasets
#     ):
#         self.data_dir = data_dir
#         self.images_dir = images_dir # Store verified images directory
#         self.split = split
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.max_length = max_length
#         self.hf_token = hf_token # Store token if provided
#         self.tokenizer = self._load_tokenizer(tokenizer_path)
#         self.examples = self._load_and_prepare_examples()

#         if not self.examples:
#             logger.warning(f"Loaded 0 examples from MathVista ({self.split} split). "
#                            f"Check dataset source, split name ('{split}'), and image directory ('{images_dir}').")
#             # Handle empty dataset case - create an empty TF dataset with correct structure
#             self.tf_dataset = self._create_empty_tf_dataset()
#         else:
#             logger.info(f"Loaded and prepared {len(self.examples)} examples for split '{self.split}'.")
#             self.tf_dataset = self._create_tf_dataset()

#     def _load_tokenizer(self, tokenizer_path: Optional[str]):
#         """Loads the SentencePiece tokenizer."""
#         if tokenizer_path and Path(tokenizer_path).exists():
#             try:
#                 tokenizer = spm.SentencePieceProcessor()
#                 tokenizer.load(tokenizer_path)
#                 logger.info(f"Tokenizer loaded from {tokenizer_path} with vocab size: {tokenizer.vocab_size()}")
#                 return tokenizer
#             except Exception as e:
#                 logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}", exc_info=True)
#                 return None
#         else:
#             logger.warning(f"Tokenizer path '{tokenizer_path}' not provided or not found. Raw text will be used.")
#             return None

#     def _load_and_prepare_examples(self) -> List[Dict]:
#         """Loads dataset metadata, verifies image paths, and formats prompts."""
#         try:
#             logger.info(f"Loading MathVista metadata for split '{self.split}'...")
#             # Pass token if provided for gated datasets
#             # Add download_mode argument to force refresh
#             dataset_metadata = load_dataset(
#                 "AI4Math/MathVista",
#                 split=self.split,
#                 token=self.hf_token,
#                 trust_remote_code=True, # Required by some datasets, use with caution
#                 download_mode="force_redownload" # Force refresh from Hub
#             )
#             logger.info(f"Raw metadata loaded with {len(dataset_metadata)} entries. Preparing examples...")
#         # --- Corrected Exception Handling ---
#         except HfHubHTTPError as e: # Catch specific Hub HTTP errors (like 401, 404)
#              logger.error(f"Hugging Face Hub HTTP Error loading metadata for split '{self.split}': {e}. "
#                           "Check authentication (huggingface-cli login), dataset name/split, and permissions.")
#              raise e
#         except requests.exceptions.RequestException as e: # Catch general network errors during download
#              logger.error(f"Network error during metadata download for split '{self.split}': {e}")
#              raise e
#         # --- End Corrected Exception Handling ---
#         except ValueError as e: # Catch the specific "Unknown split" error
#              logger.error(f"Failed to load split '{self.split}': {e}")
#              raise e
#         except Exception as e: # Catch any other unexpected errors
#             logger.error(f"Unexpected error loading MathVista metadata: {e}", exc_info=True)
#             raise e

#         prepared_examples = []
#         missing_images = 0
#         for example in tqdm(dataset_metadata, desc=f"Preparing {self.split} examples", leave=False):
#             pid = example.get("pid")
#             # Get the filename string from metadata, e.g., "images/1.jpg"
#             image_filename_with_prefix = example.get("image")

#             if not pid or not image_filename_with_prefix:
#                 logger.debug(f"Skipping example due to missing 'pid' or 'image' filename: {example}")
#                 continue

#             # --- CORRECTED PATH HANDLING ---
#             # Extract only the actual filename (part after the last '/')
#             image_basename = os.path.basename(image_filename_with_prefix) # Gets "1.jpg" from "images/1.jpg"
#             if not image_basename: # Handle cases where input might be empty or just "/"
#                  logger.warning(f"Could not extract base filename from metadata field: '{image_filename_with_prefix}' for PID {pid}. Skipping.")
#                  missing_images += 1
#                  continue

#             # Construct the full path using the images directory and the extracted basename
#             image_path = Path(self.images_dir) / image_basename
#             # --- END CORRECTION ---


#             # CRITICAL: Check if the image file actually exists *before* adding
#             if not image_path.is_file():
#                 # --- Add Enhanced Debug Logging ---
#                 if missing_images < 5: # Log details only for the first few failures
#                     # Log the path we *actually* checked
#                     logger.debug(f"Image check failed for PID {pid}. Checked path: {image_path} (derived from metadata: '{image_filename_with_prefix}')")
#                     # Try to log the first few actual entries in the images_dir for comparison
#                     if missing_images == 0:
#                         try:
#                             parent_dir_contents = list(Path(self.images_dir).glob('*'))[:10]
#                             logger.debug(f"Actual contents found directly within '{self.images_dir}': {parent_dir_contents}")
#                         except Exception as list_e:
#                             logger.debug(f"Could not list contents of '{self.images_dir}': {list_e}")
#                 # --- End Enhanced Debug Logging ---
#                 missing_images += 1
#                 continue # Skip this example

#             question = example.get("question", "").strip()
#             answer = str(example.get("answer", "")).strip() # Ensure answer is string

#             # Format the prompt specifically for Gemma 3, indicating image presence
#             gemma_prompt = format_gemma_prompt(question, image_provided=True)

#             prepared_examples.append({
#                 "id": str(pid), # Ensure ID is string
#                 "raw_prompt": question, # Keep original question if needed
#                 "answer": answer,
#                  # Store the validated, full path as string
#                 "image_path": str(image_path),
#                 "gemma_prompt": gemma_prompt, # The prompt formatted for the model
#             })

#         if missing_images > 0:
#              logger.warning(f"Skipped {missing_images} examples due to missing image files in '{self.images_dir}'.")

#         return prepared_examples

#     def _py_process_image(self, image_path_bytes):
#         """
#         Wrapper for image processing using PIL/Numpy, suitable for tf.py_function.
#         Takes bytes (TF tensor content) and returns a numpy array.
#         """
#         image_path = image_path_bytes.numpy().decode('utf-8') # Decode bytes to string path
#         img_array = resize_image_for_gemma(image_path, self.image_size)

#         if img_array is None:
#             # Handle error: return zeros or raise error if needed within py_function
#             logger.warning(f"Image processing failed for {image_path}, returning zeros.")
#             return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
#         # TF expects float32 from py_function, cast later to bfloat16 if needed
#         return img_array.astype(np.float32)


#     def process_example(self, example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
#         """
#         Processes a single example tensor dictionary within the tf.data pipeline.
#         Reads and preprocesses the image using tf.py_function.
#         Passes through text fields.
#         """
#         image_path_tensor = example["image_path"] # tf.string tensor

#         # Use tf.py_function to run the PIL/Numpy image processing
#         # Input signature: a tensor containing the string path
#         # Output signature: a float32 tensor for the image array
#         img_tensor = tf.py_function(
#             func=self._py_process_image,
#             inp=[image_path_tensor],
#             Tout=tf.float32 # Output type from the py_function
#         )

#         # Set shape explicitly as py_function loses shape information
#         img_tensor.set_shape([self.image_size, self.image_size, 3])

#         # Cast to bfloat16 after processing (better precision during processing)
#         img_tensor = tf.cast(img_tensor, dtype=tf.bfloat16)

#         # Tokenization (if tokenizer exists and needed here):
#         # Would typically involve another tf.py_function or doing it offline/later.
#         # For now, return the string tensors.
#         prompt_tensor = example["gemma_prompt"] # Use the formatted prompt
#         answer_tensor = example["answer"]
#         id_tensor = example["id"]

#         # Return the dictionary structure expected by the model/training loop
#         return {
#             "id": id_tensor,         # For tracking
#             "prompt": prompt_tensor, # The input prompt for the model
#             "image": img_tensor,     # The processed image tensor
#             "answer": answer_tensor, # The ground truth answer (string)
#             # Add 'labels' or 'decoder_input_ids' if tokenizing here
#         }


#     def _create_tf_dataset(self) -> tf.data.Dataset:
#         """Creates the TensorFlow Dataset pipeline from prepared examples."""
#         if not self.examples:
#              logger.error("Cannot create TensorFlow dataset: No valid examples were prepared.")
#              return self._create_empty_tf_dataset() # Return empty structured dataset

#         try:
#             # Structure data for from_tensor_slices: dict of lists/tensors
#             # Ensure all keys expected by from_tensor_slices exist in all examples
#             required_keys = ["id", "image_path", "gemma_prompt", "answer", "raw_prompt"]
#             structured_examples = {
#                 key: [d.get(key, "") for d in self.examples] # Use get with default for safety
#                 for key in required_keys
#             }

#             # Create dataset with explicit type specification for strings
#             dataset = tf.data.Dataset.from_tensor_slices({
#                 "id": tf.constant(structured_examples["id"], dtype=tf.string),
#                 "image_path": tf.constant(structured_examples["image_path"], dtype=tf.string),
#                 "gemma_prompt": tf.constant(structured_examples["gemma_prompt"], dtype=tf.string),
#                 "answer": tf.constant(structured_examples["answer"], dtype=tf.string),
#                 # Add other fields if they exist in self.examples (like "raw_prompt")
#                 "raw_prompt": tf.constant(structured_examples["raw_prompt"], dtype=tf.string)
#             })
#             logger.info("tf.data.Dataset created from tensor slices.")

#         except KeyError as e:
#              logger.error(f"Missing key {e} in one or more examples when structuring for tf.data.Dataset. Check data preparation.")
#              if self.examples: logger.error(f"First example keys: {list(self.examples[0].keys())}")
#              raise
#         except Exception as e:
#             logger.error(f"Failed to create tf.data.Dataset from examples: {e}", exc_info=True)
#             if self.examples: logger.error(f"First example keys: {list(self.examples[0].keys())}")
#             raise # Propagate error

#         # Apply the processing function using .map()
#         # Select only the fields needed by process_example before mapping
#         dataset_to_map = dataset.map(lambda x: {
#             "id": x["id"],
#             "image_path": x["image_path"],
#             "gemma_prompt": x["gemma_prompt"],
#             "answer": x["answer"]
#             # Note: 'raw_prompt' is filtered out here as process_example doesn't use it
#         })

#         dataset = dataset_to_map.map(
#             self.process_example,
#             num_parallel_calls=tf.data.AUTOTUNE # Enable parallel processing
#         )
#         logger.info("Processing function mapped to dataset.")

#         # Filter out potential errors if image loading failed (optional, depends on how errors are handled)
#         # dataset = dataset.filter(lambda x: tf.reduce_all(tf.shape(x['image']) > 0)) # Example filter

#         # Batch the dataset
#         dataset = dataset.batch(self.batch_size)
#         logger.info(f"Dataset batched with size {self.batch_size}.")

#         # Prefetch for performance
#         dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#         logger.info("Dataset prefetching enabled.")

#         return dataset

#     def _create_empty_tf_dataset(self) -> tf.data.Dataset:
#          """Creates an empty TF dataset with the expected structure and types."""
#          logger.warning("Creating an empty TensorFlow dataset due to lack of examples.")
#          empty_spec = { # Define the structure and types explicitly
#              "id": tf.TensorSpec(shape=(), dtype=tf.string),
#              "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
#              "gemma_prompt": tf.TensorSpec(shape=(), dtype=tf.string),
#              "answer": tf.TensorSpec(shape=(), dtype=tf.string),
#              "raw_prompt": tf.TensorSpec(shape=(), dtype=tf.string)
#          }
#          empty_dataset = tf.data.Dataset.from_generator(lambda: [], output_signature=empty_spec)

#          # Select necessary fields before mapping
#          empty_dataset_to_map = empty_dataset.map(lambda x: {
#             "id": x["id"],
#             "image_path": x["image_path"],
#             "gemma_prompt": x["gemma_prompt"],
#             "answer": x["answer"]
#          })
#          # Apply map to get the final structure, even though it will process nothing
#          empty_dataset = empty_dataset_to_map.map(self.process_example, num_parallel_calls=tf.data.AUTOTUNE)
#          empty_dataset = empty_dataset.batch(self.batch_size)
#          empty_dataset = empty_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#          return empty_dataset


#     def get_tf_dataset(self) -> tf.data.Dataset:
#         """Returns the configured TensorFlow dataset."""
#         return self.tf_dataset

#     def get_examples(self) -> List[Dict]:
#         """Returns the raw prepared example dictionaries (metadata)."""
#         return self.examples

#     def __len__(self) -> int:
#         """Returns the number of prepared examples."""
#         return len(self.examples)



