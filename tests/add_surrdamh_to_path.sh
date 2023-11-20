#!/bin/bash

# Script to add a specified folder to PYTHONPATH
# source tests/add_surrdamh_to_path.sh

# Define the folder path
FOLDER_PATH="/workspaces/surrDAMH"

# Add the folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$FOLDER_PATH

echo $PYTHONPATH