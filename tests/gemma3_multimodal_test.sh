#!/bin/bash
# gemma3_multimodal_test.sh - Templates for testing Gemma 3 multimodal inference
# Usage: 
#   1. Make executable: chmod +x gemma3_multimodal_test.sh
#   2. Run a specific example: ./gemma3_multimodal_test.sh [example_number]
#   3. Run all examples: ./gemma3_multimodal_test.sh all
#
# This script provides various templates for testing Gemma 3's multimodal capabilities

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set paths (modify as needed)
SCRIPT_PATH="./mmodal_inference_test1.py"  # Direct path to the Python file
MODEL_SIZE="4b"  # Change to 12b or 27b if you have those models

# Function to run a command with header
run_example() {
    NUMBER=$1
    TITLE=$2
    CMD=$3
    
    echo -e "\n${GREEN}========================================================================"
    echo -e "EXAMPLE $NUMBER: $TITLE"
    echo -e "========================================================================${NC}"
    echo -e "${BLUE}Running: $CMD${NC}\n"
    eval $CMD
    echo -e "\n${GREEN}========================================================================"
    echo -e "END OF EXAMPLE $NUMBER"
    echo -e "========================================================================${NC}"
    
    # Small pause between examples
    if [[ "$4" != "no_pause" ]]; then
        echo "Press enter to continue..."
        read
    fi
}

# Parse arguments
EXAMPLE=$1
if [[ -z "$EXAMPLE" ]]; then
    echo "Please specify an example number (1-10) or 'all'"
    exit 1
fi

# Example 1: Basic image description with auto-formatted prompt
if [[ "$EXAMPLE" == "1" || "$EXAMPLE" == "all" ]]; then
    run_example 1 "Basic Image Description" \
    "python $SCRIPT_PATH \
        --question \"What can you see in this image? Describe it in detail.\" \
        --model_size $MODEL_SIZE"
fi

# Example 2: Mathematics problem solving
if [[ "$EXAMPLE" == "2" || "$EXAMPLE" == "all" ]]; then
    run_example 2 "Math Problem Analysis" \
    "python $SCRIPT_PATH \
        --question \"This image contains a mathematical problem. Solve it step-by-step and explain your reasoning.\" \
        --model_size $MODEL_SIZE \
        --max_tokens 512"
fi

# Example 3: Custom prompt with temperature adjustment
if [[ "$EXAMPLE" == "3" || "$EXAMPLE" == "all" ]]; then
    run_example 3 "Custom Prompt with Temperature" \
    "python $SCRIPT_PATH \
        --prompt \"<start_of_turn>user\\nExplain what's happening in this image as if you were a professional photographer analyzing the composition.\\n<start_of_image>\\n<end_of_turn>\\n<start_of_turn>model\" \
        --temperature 0.9 \
        --model_size $MODEL_SIZE"
fi

# Example 4: Object identification with top-p and top-k adjustments
if [[ "$EXAMPLE" == "4" || "$EXAMPLE" == "all" ]]; then
    run_example 4 "Object Identification" \
    "python $SCRIPT_PATH \
        --question \"Identify all objects in this image and their approximate positions.\" \
        --top_p 0.92 \
        --top_k 60 \
        --model_size $MODEL_SIZE"
fi

# Example 5: Detailed analysis with more tokens
if [[ "$EXAMPLE" == "5" || "$EXAMPLE" == "all" ]]; then
    run_example 5 "Detailed Analysis" \
    "python $SCRIPT_PATH \
        --question \"Analyze this image in extreme detail. Describe colors, objects, composition, lighting, and any text visible.\" \
        --max_tokens 1024 \
        --model_size $MODEL_SIZE"
fi

# Example 6: Specific image with custom prompt
if [[ "$EXAMPLE" == "6" || "$EXAMPLE" == "all" ]]; then
    run_example 6 "Specific Image" \
    "python $SCRIPT_PATH \
        --question \"What do you see in this image?\" \
        --image \"data/images/test_batch_image.png\" \
        --model_size $MODEL_SIZE"
fi

# Example 7: Math problem with step-by-step reasoning
if [[ "$EXAMPLE" == "7" || "$EXAMPLE" == "all" ]]; then
    run_example 7 "Math Reasoning" \
    "python $SCRIPT_PATH \
        --prompt \"<start_of_turn>user\\nSolve this mathematical problem step-by-step and give me all the reasoning traces. Format your final answer as <answer>YOUR ANSWER</answer>\\n<start_of_image>\\n<end_of_turn>\\n<start_of_turn>model\" \
        --max_tokens -1 \
        --model_size $MODEL_SIZE"
fi

# Example 8: Comparison of objects
if [[ "$EXAMPLE" == "8" || "$EXAMPLE" == "all" ]]; then
    run_example 8 "Comparison Analysis" \
    "python $SCRIPT_PATH \
        --question \"If there are multiple objects in this image, compare and contrast them.\" \
        --model_size $MODEL_SIZE"
fi

# Example 9: Creative writing based on image
if [[ "$EXAMPLE" == "9" || "$EXAMPLE" == "all" ]]; then
    run_example 9 "Creative Writing" \
    "python $SCRIPT_PATH \
        --question \"Write a short creative story (around 200 words) inspired by this image.\" \
        --temperature 0.9 \
        --top_p 0.95 \
        --model_size $MODEL_SIZE \
        --max_tokens 1024"
fi

# Example 10: Technical analysis
if [[ "$EXAMPLE" == "10" || "$EXAMPLE" == "all" ]]; then
    run_example 10 "Technical Analysis" \
    "python $SCRIPT_PATH \
        --question \"Provide a technical analysis of what's shown in this image. If it contains any charts, graphs, or technical elements, explain them in detail.\" \
        --max_tokens 768 \
        --model_size $MODEL_SIZE"
fi

# MathVista specific examples
if [[ "$EXAMPLE" == "11" || "$EXAMPLE" == "all" ]]; then
    run_example 11 "MathVista Basic" \
    "python $SCRIPT_PATH \
        --prompt \"<start_of_turn>user\\nSolve this mathematical problem. Calculate step by step and provide your final answer in the format: <answer>YOUR ANSWER</answer>\\n<start_of_image>\\n<end_of_turn>\\n<start_of_turn>model\" \
        --max_tokens 1024 \
        --model_size $MODEL_SIZE"
fi

if [[ "$EXAMPLE" == "12" || "$EXAMPLE" == "all" ]]; then
    run_example 12 "MathVista Geometry" \
    "python $SCRIPT_PATH \
        --prompt \"<start_of_turn>user\\nThis is a geometry problem. Analyze the diagram, identify the relevant geometric principles, and solve step by step. Provide your final answer in the format: <answer>YOUR ANSWER</answer>\\n<start_of_image>\\n<end_of_turn>\\n<start_of_turn>model\" \
        --max_tokens 1024 \
        --temperature 0.3 \
        --model_size $MODEL_SIZE"
fi

if [[ "$EXAMPLE" == "13" || "$EXAMPLE" == "all" ]]; then
    run_example 13 "MathVista Graph Interpretation" \
    "python $SCRIPT_PATH \
        --prompt \"<start_of_turn>user\\nThis image contains a graph or chart. Carefully analyze the data visualization, extract the key information, and answer any questions posed. Provide your final answer in the format: <answer>YOUR ANSWER</answer>\\n<start_of_image>\\n<end_of_turn>\\n<start_of_turn>model\" \
        --max_tokens 1024 \
        --model_size $MODEL_SIZE"
fi

echo -e "\n${GREEN}All requested examples completed!${NC}"