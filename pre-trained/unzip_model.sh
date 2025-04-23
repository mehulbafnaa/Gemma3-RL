#!/usr/bin/env bash

# Simple script to extract a .tar.gz archive file
# Usage: ./extract_archive.sh <filename.tar.gz>

# --- Get Input ---
# The archive file is the first argument passed to the script
ARCHIVE_FILE="$1"

# --- Validation ---
# Check if an argument was provided
if [ -z "$ARCHIVE_FILE" ]; then
  echo "Error: No archive filename provided."
  echo "Usage: $0 <filename.tar.gz>"
  exit 1
fi

# Check if the specified file exists
if [ ! -f "$ARCHIVE_FILE" ]; then
  echo "Error: File '$ARCHIVE_FILE' not found."
  exit 1
fi

# Optional: Basic check if it ends with .tar.gz (can be removed if needed)
if [[ "$ARCHIVE_FILE" != *.tar.gz ]]; then
   echo "Warning: Filename '$ARCHIVE_FILE' does not end with .tar.gz."
   # Decide if you want to proceed anyway or exit
   # exit 1 
fi

# --- Extraction ---
echo "Attempting to extract '$ARCHIVE_FILE'..."

# Use tar command:
# -x : Extract files from an archive
# -z : Filter the archive through gzip (decompress .gz)
# -v : Verbose - list files processed (optional, but helpful)
# -f : Use archive file (this MUST be followed immediately by the filename)
tar -xzvf "$ARCHIVE_FILE"

# Check the exit status of the tar command
# $? holds the exit status of the last command (0 means success)
if [ $? -eq 0 ]; then
  echo "Extraction of '$ARCHIVE_FILE' completed successfully."
  exit 0
else
  echo "Error occurred during extraction of '$ARCHIVE_FILE'."
  exit 1
fi