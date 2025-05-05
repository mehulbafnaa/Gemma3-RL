"""
Checkpoint utilities for saving and loading model state with Orbax.

This module provides functions to save and load model checkpoints using
the orbax-checkpoint library, which is specifically designed for JAX
models and optimizers.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_checkpoint_manager(
    checkpoint_dir: str,
    max_to_keep: int = 5,
    create: bool = True
) -> ocp.CheckpointManager:
    """
    Initialize an Orbax checkpoint manager.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Configured CheckpointManager instance
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    if create:
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Configure the checkpointer options
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=create
    )
    
    # Create and return the checkpoint manager
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(checkpoint_dir_path),
        checkpointers={'model': checkpointer},
        options=options
    )
    
    logger.info(f"Initialized checkpoint manager at: {checkpoint_dir}")
    return checkpoint_manager

def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    state: Dict[str, Any],
    step: int,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a checkpoint using the checkpoint manager.
    
    Args:
        checkpoint_manager: The checkpoint manager to use
        state: Dictionary containing model state
        step: Training step number
        metadata: Optional metadata to save with checkpoint
        
    Returns:
        Path to the saved checkpoint
    """
    try:
        # Save the checkpoint
        save_args = ocp.args.StandardSave(state)
        checkpoint_manager.save(
            step,
            args={'model': save_args},
            metrics=metadata
        )
        logger.info(f"Saved checkpoint at step {step}")
        return os.path.join(checkpoint_manager.directory, f"{step}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {step}: {e}")
        raise

def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Optional specific step to load, if None loads latest
        
    Returns:
        Dictionary with loaded model state
    """
    try:
        checkpointer = ocp.StandardCheckpointer()
        
        # Determine the step to load
        if step is None:
            # Find the latest checkpoint
            checkpoint_manager = initialize_checkpoint_manager(
                checkpoint_dir=checkpoint_dir,
                create=False
            )
            latest_step = checkpoint_manager.latest_step()
            if latest_step is None:
                raise ValueError(f"No checkpoints found in {checkpoint_dir}")
            step = latest_step
            logger.info(f"Loading latest checkpoint at step {step}")
        else:
            logger.info(f"Loading checkpoint at specified step {step}")
        
        # Load the checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, str(step))
        state_dict = checkpointer.restore(checkpoint_path)
        
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def save_model_state(
    checkpoint_dir: str,
    model_params: Dict,
    step: int,
    tokenizer_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_to_keep: int = 5
) -> str:
    """
    Save model parameters and associated data.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model_params: Model parameters to save
        step: Training step number
        tokenizer_path: Optional path to tokenizer
        metadata: Optional metadata to save
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Path to the saved checkpoint
    """
    # Create state dictionary to save
    state = {
        'params': model_params,
    }
    
    # Add metadata
    if metadata is None:
        metadata = {}
    
    if tokenizer_path:
        metadata['tokenizer_path'] = tokenizer_path
    
    # Initialize checkpoint manager
    checkpoint_manager = initialize_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep
    )
    
    # Save checkpoint
    return save_checkpoint(
        checkpoint_manager=checkpoint_manager,
        state=state,
        step=step,
        metadata=metadata
    )

def load_model_state(
    checkpoint_dir: str,
    step: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """
    Load model parameters and associated data.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Optional specific step to load, if None loads latest
        
    Returns:
        Tuple of (model_params, metadata)
    """
    # Load checkpoint
    state_dict = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step
    )
    
    # Extract model parameters and metadata
    model_params = state_dict.get('params', {})
    metadata = {}
    
    for key, value in state_dict.items():
        if key != 'params':
            metadata[key] = value
    
    return model_params, metadata

def save_train_state(
    checkpoint_dir: str,
    train_state: train_state.TrainState,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_to_keep: int = 5
) -> str:
    """
    Save a Flax TrainState object.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        train_state: Flax TrainState to save
        step: Training step (if None, uses train_state.step)
        metadata: Optional metadata to save
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Path to the saved checkpoint
    """
    if step is None:
        step = int(train_state.step)
    
    # Create state dictionary to save
    state = {
        'params': train_state.params,
        'opt_state': train_state.opt_state,
        'step': step,
    }
    
    # Initialize checkpoint manager
    checkpoint_manager = initialize_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep
    )
    
    # Save checkpoint
    return save_checkpoint(
        checkpoint_manager=checkpoint_manager,
        state=state,
        step=step,
        metadata=metadata
    )

def load_train_state(
    checkpoint_dir: str,
    train_state_class: Any,
    step: Optional[int] = None
) -> Tuple[train_state.TrainState, Dict]:
    """
    Load a Flax TrainState object.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_state_class: Class or factory function to create train state
        step: Optional specific step to load, if None loads latest
        
    Returns:
        Tuple of (train_state, metadata)
    """
    # Load checkpoint
    state_dict = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step
    )
    
    # Extract train state components and metadata
    params = state_dict.get('params', {})
    opt_state = state_dict.get('opt_state', None)
    step_value = state_dict.get('step', 0)
    
    metadata = {}
    for key, value in state_dict.items():
        if key not in ['params', 'opt_state', 'step']:
            metadata[key] = value
    
    # Recreate train state
    if hasattr(train_state_class, 'restore'):
        # If the class has a custom restore method
        restored_train_state = train_state_class.restore(
            params=params,
            opt_state=opt_state,
            step=step_value
        )
    else:
        # Default reconstruction
        restored_train_state = train_state.TrainState(
            step=step_value,
            params=params,
            opt_state=opt_state,
            apply_fn=train_state_class.apply_fn,
            tx=train_state_class.tx
        )
    
    return restored_train_state, metadata

def get_latest_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    """
    Get the latest checkpoint step from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Latest step number or None if no checkpoints found
    """
    try:
        checkpoint_manager = initialize_checkpoint_manager(
            checkpoint_dir=checkpoint_dir,
            create=False
        )
        return checkpoint_manager.latest_step()
    except Exception as e:
        logger.error(f"Failed to get latest checkpoint step: {e}")
        return None

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create dummy params for testing
    params = {"weight": jnp.array(np.random.random((10, 10)))}
    
    # Save checkpoint
    save_dir = "/tmp/test_checkpoint"
    save_model_state(
        checkpoint_dir=save_dir,
        model_params=params,
        step=0,
        metadata={"test": "metadata"}
    )
    
    # Load checkpoint
    loaded_params, loaded_metadata = load_model_state(
        checkpoint_dir=save_dir
    )
    
    print(f"Loaded parameters shape: {loaded_params['weight'].shape}")
    print(f"Loaded metadata: {loaded_metadata}")