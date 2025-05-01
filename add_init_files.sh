#!/bin/bash
# add_init_files.sh - Add __init__.py files to Python package directories
# Usage: ./add_init_files.sh [--dry-run] [path]
#   --dry-run: Show which directories would be modified without making changes
#   path: Starting directory (defaults to current directory)

set -e  # Exit on error

# Parse arguments
DRY_RUN=0
START_PATH="."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            START_PATH="$1"
            shift
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Looking for Python package directories in: ${START_PATH}${NC}"

# Find directories containing .py files but no __init__.py
echo "Scanning directories..."
DIRS_TO_FIX=()

# Recursive function to check directories
check_directory() {
    local dir="$1"
    local has_py_files=0
    local has_init_py=0
    
    # Skip hidden directories and virtual environments
    if [[ $(basename "$dir") == .* || "$dir" == *"venv"* || "$dir" == *"__pycache__"* ]]; then
        return
    fi
    
    # Check if directory contains .py files
    if ls "$dir"/*.py 1> /dev/null 2>&1; then
        has_py_files=1
    fi
    
    # Check if directory already has __init__.py
    if [ -f "$dir/__init__.py" ]; then
        has_init_py=1
    fi
    
    # If there are .py files but no __init__.py, add to fix list
    if [ $has_py_files -eq 1 ] && [ $has_init_py -eq 0 ]; then
        DIRS_TO_FIX+=("$dir")
    fi
    
    # Recursively check subdirectories
    for subdir in "$dir"/*/; do
        if [ -d "$subdir" ]; then
            check_directory "$subdir"
        fi
    done
}

# Start the recursive checking
check_directory "$START_PATH"

# Report and create __init__.py files
if [ ${#DIRS_TO_FIX[@]} -eq 0 ]; then
    echo -e "${GREEN}All Python package directories already have __init__.py files.${NC}"
else
    echo -e "${YELLOW}Found ${#DIRS_TO_FIX[@]} directories that need __init__.py files:${NC}"
    for dir in "${DIRS_TO_FIX[@]}"; do
        echo "  $dir"
        
        if [ $DRY_RUN -eq 0 ]; then
            # Create the file with a simple header comment
            echo "# -*- coding: utf-8 -*-" > "$dir/__init__.py"
            echo "# Auto-generated __init__.py file" >> "$dir/__init__.py"
            echo -e "${GREEN}Created: $dir/__init__.py${NC}"
        fi
    done
    
    if [ $DRY_RUN -eq 1 ]; then
        echo -e "${YELLOW}DRY RUN: No files were created. Run without --dry-run to create the files.${NC}"
    else
        echo -e "${GREEN}Created ${#DIRS_TO_FIX[@]} __init__.py files.${NC}"
    fi
fi

# Check specific project directories that should always have __init__.py
IMPORTANT_DIRS=("tests" "src" "src/data" "src/model" "src/grpo" "src/reward" "src/utils")
MISSING_IMPORTANT=0

echo -e "\n${GREEN}Checking important project directories:${NC}"
for dir in "${IMPORTANT_DIRS[@]}"; do
    if [ -d "$START_PATH/$dir" ]; then
        if [ ! -f "$START_PATH/$dir/__init__.py" ]; then
            echo -e "${YELLOW}WARNING: $dir/ exists but has no __init__.py file${NC}"
            MISSING_IMPORTANT=1
            
            if [ $DRY_RUN -eq 0 ]; then
                # Create the file with a proper package header
                echo "# -*- coding: utf-8 -*-" > "$START_PATH/$dir/__init__.py"
                echo "# Auto-generated __init__.py file for $dir package" >> "$START_PATH/$dir/__init__.py"
                echo -e "${GREEN}Created: $START_PATH/$dir/__init__.py${NC}"
            fi
        else
            echo -e "${GREEN}✓ $dir/ has __init__.py${NC}"
        fi
    else
        echo -e "${YELLOW}Note: $dir/ not found (may not exist in your project)${NC}"
    fi
done

if [ $MISSING_IMPORTANT -eq 1 ] && [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}DRY RUN: Important directories are missing __init__.py files. Run without --dry-run to create them.${NC}"
fi

echo -e "\n${GREEN}Initialization file check complete!${NC}"