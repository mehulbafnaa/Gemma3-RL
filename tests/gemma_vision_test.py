# tests/model_test.py

import unittest
import os
import sys
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock, PropertyMock # Ensure PropertyMock is imported

# Ensure src directory is in path (assuming tests are run from project root)
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import the class we want to test
try:
    from src.model.gemma_vision import GemmaVisionModel
    # Import constants using the fallback defined in the module itself
    from src.model.gemma_vision import BOS_TOKEN, START_OF_TURN, END_OF_TURN, USER, MODEL, IMAGE_TOKEN
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("Ensure you run tests from the project root directory.")
    sys.exit(1)

# --- Dummy Files Setup ---
DUMMY_CONFIG_PATH = Path("./dummy_config.yaml")
DUMMY_TOKENIZER_PATH = Path("./dummy_tokenizer.model") # Relative path for test
DUMMY_PRESET_PATH = Path("./dummy_preset")

# Dummy config content
DUMMY_CONFIG = {
    "model": {
        "preset_path": str(DUMMY_PRESET_PATH.name), # Use relative name for config content
    },
    "train": {
        "mixed_precision_policy": "mixed_bfloat16"
    }
}

# --- Mock SentencePieceProcessor globally for tests ---
mock_spm_processor = MagicMock(name="MockSPMProcessor") # Added name for clarity
mock_spm_processor.vocab_size = 1000
mock_spm_processor.bos_id = 2
mock_spm_processor.eos_id = 1
mock_spm_processor.pad_id = 0
# Use lambda with default args captured correctly
mock_spm_processor.encode.side_effect = lambda s, add_bos=False, add_eos=False, mock=mock_spm_processor: [mock.bos_id]*add_bos + [ord(c) for c in s] + [mock.eos_id]*add_eos
mock_spm_processor.decode.side_effect = lambda ids, mock=mock_spm_processor: "".join([chr(i) for i in ids if i not in [mock.bos_id, mock.eos_id, mock.pad_id]]) # Ignore special tokens

mock_spm_class = MagicMock(name="MockSPMClass")
mock_spm_class.SentencePieceProcessor.return_value = mock_spm_processor
sys.modules['sentencepiece'] = mock_spm_class # Ensure spm is mocked


class TestGemmaVisionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create dummy files once for the whole class."""
        print("\nSetting up dummy files for test class...")
        DUMMY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DUMMY_CONFIG_PATH, 'w') as f:
            yaml.dump(DUMMY_CONFIG, f)
        DUMMY_PRESET_PATH.mkdir(exist_ok=True)
        with open(DUMMY_TOKENIZER_PATH, 'w') as f: f.write("")

    @classmethod
    def tearDownClass(cls):
        """Remove dummy files once after all tests."""
        print("\nTearing down dummy files for test class...")
        if DUMMY_CONFIG_PATH.exists(): DUMMY_CONFIG_PATH.unlink()
        if DUMMY_TOKENIZER_PATH.exists(): DUMMY_TOKENIZER_PATH.unlink()
        if DUMMY_PRESET_PATH.exists():
            try: DUMMY_PRESET_PATH.rmdir()
            except OSError: print(f"Warning: Could not remove dummy preset dir {DUMMY_PRESET_PATH}")

    def setUp(self):
        """Set up mocks or instance variables before each test if needed."""
        self.project_root = Path(__file__).parent.parent.resolve()

    def test_01_initialization_paths_config(self):
        """Test if the class initializes, loads config, and sets paths correctly."""
        print("Running test_01_initialization_paths_config...")
        # Patch the actual model loading during init, we only test paths here
        with patch.object(GemmaVisionModel, '_load_pretrained_model_with_tokenizer', return_value=None) as mock_load:
            model_wrapper = GemmaVisionModel(config_path=str(DUMMY_CONFIG_PATH))

            self.assertIsNotNone(model_wrapper.config, "Config should be loaded")
            self.assertEqual(model_wrapper.config.get("train", {}).get("mixed_precision_policy"), "mixed_bfloat16")
            mock_load.assert_called_once() # Check that loading was attempted

            expected_preset_path = self.project_root / DUMMY_CONFIG["model"]["preset_path"]
            self.assertEqual(Path(model_wrapper.preset_path).resolve(), expected_preset_path.resolve())
            print("Initialization paths and config loading test passed.")


    # --- CORRECTED: Mock the internal loading method ---
    @patch('src.model.gemma_vision.GemmaVisionModel._load_pretrained_model_with_tokenizer')
    def test_02_successful_load_and_tokenizer_access(self, mock_load_method):
        """Test successful load simulation and tokenizer access."""
        print("Running test_02_successful_load_and_tokenizer_access...")
        # --- Setup Mocks ---
        mock_gemma_model = MagicMock(name="MockGemmaModel", spec=['tokenizer', 'generate', 'call'])
        mock_tokenizer_instance = mock_spm_processor # Use the globally mocked instance
        type(mock_gemma_model).tokenizer = PropertyMock(return_value=mock_tokenizer_instance)
        mock_load_method.return_value = mock_gemma_model # Configure the mock method

        # --- Instantiate and Test ---
        model_wrapper = GemmaVisionModel(config_path=str(DUMMY_CONFIG_PATH))

        # Assertions
        mock_load_method.assert_called_once()
        self.assertIsNotNone(model_wrapper.gemma_model, "gemma_model should be set")
        self.assertIs(model_wrapper.gemma_model, mock_gemma_model, "gemma_model should be the mocked object")
        self.assertIsNotNone(model_wrapper.tokenizer, "tokenizer should be set from loaded model")
        self.assertIs(model_wrapper.tokenizer, mock_tokenizer_instance, "tokenizer should be the mocked object")
        self.assertEqual(model_wrapper.tokenizer.vocab_size, 1000)
        print("Successful load simulation and tokenizer access test passed.")

    # --- CORRECTED: Mock the internal loading method ---
    @patch('src.model.gemma_vision.GemmaVisionModel._load_pretrained_model_with_tokenizer')
    def test_03_generate_success_mocked(self, mock_load_method):
        """Test the generate method with a mocked model and tokenizer."""
        print("Running test_03_generate_success_mocked...")
        # --- Setup Mocks ---
        mock_gemma_model = MagicMock(name="MockGemmaModelGenerate", spec=['tokenizer', 'generate', 'call'])
        mock_tokenizer_instance = mock_spm_processor
        type(mock_gemma_model).tokenizer = PropertyMock(return_value=mock_tokenizer_instance)

        # Configure the mock model's generate method
        dummy_output_tokens = [ord(' '), ord('G'), ord('e'), ord('n'), ord('!'), mock_tokenizer_instance.eos_id]
        def mock_generate_func(*args, **kwargs):
            # Assume generate gets called with batch_input_tokens (list of lists)
            # Simulate returning prompt + new tokens
            # The actual model generate might take kwargs like max_length etc.
            # We base return on the first positional arg assumed to be tokens/prompts
            input_arg = args[0]
            # Simplified: assume input is list of token lists from internal processing
            # This mock needs to align with how generate *actually calls* the underlying model
            # For now, let's assume it passes the list of tokenized prompts from step 2 inside generate
            # which we mocked earlier with simple ord()
            processed_prompts = kwargs.get('prompts') # Check if prompts passed as kwarg
            if processed_prompts:
                 tokenized_prompts = [[ord(c) for c in p] for p in processed_prompts]
                 return [p_tokens + dummy_output_tokens for p_tokens in tokenized_prompts]
            else:
                 # Fallback if called differently (less likely with wrapper)
                 return [dummy_output_tokens] * len(input_arg if isinstance(input_arg, list) else [input_arg])


        # Assign mock generate method
        mock_gemma_model.generate = MagicMock(side_effect=mock_generate_func)

        # Configure the mocked loader to return the mock model
        mock_load_method.return_value = mock_gemma_model

        # --- Instantiate and Test ---
        model_wrapper = GemmaVisionModel(config_path=str(DUMMY_CONFIG_PATH))
        self.assertIsNotNone(model_wrapper.gemma_model)
        self.assertIsNotNone(model_wrapper.tokenizer)

        # --- Call generate ---
        test_prompt = "Test prompt"
        dummy_image = tf.zeros((896, 896, 3), dtype=tf.bfloat16) # Corrected image shape for single image

        generated_text = model_wrapper.generate(
            prompts=test_prompt,
            images=dummy_image, # Pass single dummy image
            max_length=50
        )

        # --- Assertions ---
        mock_gemma_model.generate.assert_called_once()

        # Check the decoded output based on mocks
        expected_output = " Gen!" # Mock tokenizer decode removes EOS and special tokens
        self.assertEqual(generated_text, expected_output)
        print("Generate success (mocked) test passed.")


    @patch('src.model.gemma_vision.GemmaVisionModel._load_pretrained_model_with_tokenizer', return_value=None)
    def test_04_generate_load_failure(self, mock_load):
        """Test generate behavior when model loading fails."""
        print("Running test_04_generate_load_failure...")
        # Instantiate - loading is mocked to return None
        model_wrapper = GemmaVisionModel(config_path=str(DUMMY_CONFIG_PATH))

        self.assertIsNone(model_wrapper.gemma_model, "Model should be None after mocked load failure")
        self.assertIsNone(model_wrapper.tokenizer, "Tokenizer should be None if model failed load")

        # Call generate
        output = model_wrapper.generate(prompts="Test")

        # --- CORRECTED ASSERTION ---
        # Assert specific list content for single prompt input
        self.assertEqual(output, ["Error: Model/Tokenizer not loaded."])
        # --- END CORRECTED ASSERTION ---
        print("Generate load failure test passed.")


if __name__ == '__main__':
    # Ensure dummy files are created relative to the test script's execution path
    try:
        os.chdir(script_dir) # Change CWD to test dir for relative paths to work
        print(f"Changed CWD to: {os.getcwd()}")
    except FileNotFoundError:
         print(f"Warning: Could not change CWD to {script_dir}. Ensure tests are run from project root or adjust dummy file paths.")

    unittest.main(verbosity=2)