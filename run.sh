#!/bin/bash
# run.sh - Main script for running Gemma3-RL training and evaluation
# Usage: ./run.sh [command] [options]

set -e  # Exit on error

# Define paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${REPO_ROOT}/configs"
DATA_DIR="${REPO_ROOT}/data"
PRETRAINED_DIR="${REPO_ROOT}/pre-trained"
LOGDIR="${REPO_ROOT}/logs"

# Create directories if they don't exist
mkdir -p "${DATA_DIR}" "${LOGDIR}"

# Ensure pre-trained directory exists
if [ ! -d "${PRETRAINED_DIR}" ]; then
    mkdir -p "${PRETRAINED_DIR}"
    echo "Created pre-trained directory at ${PRETRAINED_DIR}"
    echo "You'll need to download the Gemma3 model checkpoints to this directory."
fi

# Help message
function show_help {
    echo "Gemma3-RL: Reinforcement Learning for Gemma3 Multimodal Models"
    echo ""
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train           Train a model using GRPO"
    echo "  eval            Evaluate a model"
    echo "  test            Run model test"
    echo "  download        Download pre-trained models and datasets"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train --config configs/grpo_train_config.yaml"
    echo "  ./run.sh eval --model-path pre-trained/gemma3-4b-rl --image data/images/test.jpg"
    echo "  ./run.sh test --model-path pre-trained/gemma3-4b --prompt 'Describe this image'"
    echo ""
    echo "For more information, see the README.md file."
}

# Download pre-trained models and datasets
function download {
    echo "=== Downloading pre-trained models and datasets ==="
    
    # Define which model to download
    model_size=${1:-"4b"}
    
    case "$model_size" in
        4b)
            echo "Downloading Gemma3-4B model..."
            # This is a placeholder - you'll need to replace with actual download commands
            echo "NOTE: You need to implement the actual download or obtain the model from official sources."
            ;;
        *)
            echo "Unsupported model size: ${model_size}"
            echo "Supported sizes: 4b"
            exit 1
            ;;
    esac
    
    echo "Downloading MathVista dataset..."
    # This is a placeholder - implement actual dataset download
    python -c "from datasets import load_dataset; load_dataset('AI4Math/MathVista', 'testmini', trust_remote_code=True)"
    
    echo "Download complete!"
}

# Run training
function train {
    echo "=== Starting Gemma3-RL training ==="
    
    # Parse options
    config_file="${CONFIGS_DIR}/grpo_train_config.yaml"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                config_file="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if config file exists
    if [ ! -f "${config_file}" ]; then
        echo "Config file not found: ${config_file}"
        exit 1
    fi
    
    echo "Using config: ${config_file}"
    
    # Run training
    python "${REPO_ROOT}/run_grpo.py" \
        --config "${config_file}" \
        --output_dir "${LOGDIR}/$(date +%Y%m%d_%H%M%S)"
    
    echo "Training complete!"
}

# Run evaluation
function eval {
    echo "=== Starting model evaluation ==="
    
    # Parse options
    model_path="${PRETRAINED_DIR}/gemma3-4b"
    image_path=""
    prompt="Describe this image."
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-path)
                model_path="$2"
                shift 2
                ;;
            --image)
                image_path="$2"
                shift 2
                ;;
            --prompt)
                prompt="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if model path exists
    if [ ! -d "${model_path}" ]; then
        echo "Model path not found: ${model_path}"
        exit 1
    fi
    
    # Check if image path exists if provided
    if [ -n "${image_path}" ] && [ ! -f "${image_path}" ]; then
        echo "Image file not found: ${image_path}"
        exit 1
    fi
    
    # Construct command
    cmd="python ${REPO_ROOT}/tests/mmodal_inference_test1.py"
    
    if [ -n "${image_path}" ]; then
        cmd="${cmd} --image ${image_path}"
    fi
    
    cmd="${cmd} --prompt \"${prompt}\" --checkpoint_dir ${model_path}"
    
    # Run evaluation
    eval "${cmd}"
    
    echo "Evaluation complete!"
}

# Run model test
function test {
    echo "=== Running model test ==="
    
    # Parse options
    model_path="${PRETRAINED_DIR}/gemma3-4b"
    prompt="Test prompt"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-path)
                model_path="$2"
                shift 2
                ;;
            --prompt)
                prompt="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if model path exists
    if [ ! -d "${model_path}" ]; then
        echo "Model path not found: ${model_path}"
        exit 1
    fi
    
    # Run test
    bash "${REPO_ROOT}/tests/gemma3_multimodal_test.sh" "${model_path}" "${prompt}"
    
    echo "Test complete!"
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

command="$1"
shift

case "$command" in
    train)
        train "$@"
        ;;
    eval)
        eval "$@"
        ;;
    test)
        test "$@"
        ;;
    download)
        download "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $command"
        show_help
        exit 1
        ;;
esac