# explore_mathvista.py
import logging
from pathlib import Path
from datasets import load_dataset
import json
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_mathvista_dataset():
    """Explore and print the structure of the MathVista dataset"""
    try:
        # Load MathVista dataset
        logger.info("Loading MathVista dataset...")
        dataset = load_dataset("AI4Math/MathVista", split="test")
        
        # Print dataset info
        logger.info(f"Dataset size: {len(dataset)} examples")
        logger.info(f"Dataset features: {dataset.features}")
        
        # Get a sample item
        sample_item = dataset[0]
        logger.info("Sample item keys:")
        for key in sample_item:
            logger.info(f"  - {key}: {type(sample_item[key])}")
        
        # Print the first example in detail
        logger.info("\nFirst example details:")
        for key, value in sample_item.items():
            if key == "image":
                logger.info(f"  - {key}: Image path: {value}")
            elif isinstance(value, (dict, list)) and len(str(value)) > 100:
                logger.info(f"  - {key}: {type(value)} (truncated)")
                truncated = str(value)[:100] + "..."
                logger.info(f"    {truncated}")
            else:
                logger.info(f"  - {key}: {value}")
        
        # Check if there are any examples without images
        no_image_count = sum(1 for item in dataset if "image" not in item or not item["image"])
        logger.info(f"\nExamples without images: {no_image_count}")
        
        # Check for answer formats
        logger.info("\nExploring answer formats:")
        answer_samples = [dataset[i].get("answer", "") for i in range(min(5, len(dataset)))]
        for i, answer in enumerate(answer_samples):
            logger.info(f"Answer {i+1}: {answer}")
        
        # Print a few questions
        logger.info("\nSample questions:")
        for i in range(min(3, len(dataset))):
            logger.info(f"Question {i+1}: {dataset[i].get('question', '')}")
        
        # Check unique problem types
        if "problem_type" in sample_item:
            problem_types = set(item.get("problem_type", "") for item in dataset)
            logger.info(f"\nUnique problem types: {problem_types}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error exploring dataset: {e}")
        return None

if __name__ == "__main__":
    dataset = explore_mathvista_dataset()
    
    # Save a few examples to a JSON file for reference
    try:
        sample_size = min(5, len(dataset)) if dataset else 0
        if sample_size > 0:
            samples = [dataset[i] for i in range(sample_size)]
            
            # Convert samples to serializable format
            serializable_samples = []
            for sample in samples:
                serializable_sample = {}
                for key, value in sample.items():
                    if key == "image":
                        serializable_sample[key] = str(value)
                    else:
                        serializable_sample[key] = value
                serializable_samples.append(serializable_sample)
            
            with open("mathvista_samples.json", "w") as f:
                json.dump(serializable_samples, f, indent=2)
            logger.info(f"Saved {sample_size} samples to mathvista_samples.json")
    except Exception as e:
        logger.error(f"Error saving samples: {e}")