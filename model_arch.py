import orbax.checkpoint as ocp
import os

# Convert to absolute path
checkpoint_dir = os.path.abspath('pre-trained/gemma3-4b')
print(f"Loading from: {checkpoint_dir}")

# Load checkpoint
checkpointer = ocp.StandardCheckpointer()
checkpoint = checkpointer.restore(checkpoint_dir)

# Recursive function to print the entire structure
def print_nested(obj, path=""):
    if hasattr(obj, 'shape'):
        print(f"{path}: {obj.shape}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}/{key}" if path else key
            print_nested(value, new_path)
    else:
        print(f"{path}: {type(obj)}")

# Print everything
print("\nALL PARAMETERS:")
print_nested(checkpoint)