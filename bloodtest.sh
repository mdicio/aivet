#!/bin/bash

# Step 1: Activate the first pyenv environment and run the first model
echo "Activating the first pyenv environment..."
pyenv activate aivet  # Replace with the name of your first pyenv environment

echo "Running the first model (Image to Text extraction)..."
python bt1.py  # Modify with the path to your first Python script

# Step 2: Clear GPU memory and deactivate the first environment
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"

# Step 3: Deactivate the first environment
pyenv deactivate

# Step 4: Activate the second pyenv environment and run the second model
echo "Activating the second pyenv environment..."
pyenv activate llama32  # Replace with the name of your second pyenv environment

echo "Running the second model (Text analysis)..."
python bt2.py  # Modify with the path to your second Python script

# Step 5: Clear GPU memory and deactivate the second environment
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"

# Step 6: Deactivate the second environment
pyenv deactivate

echo "Fast diagnosis completed."
