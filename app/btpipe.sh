#!/bin/bash

# Initialize pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Get the uploaded image path from the argument
input_image=$1
echo input image $input_image
output_image_dir="app/static/uploads"
output_text_file="data/output/extracted_text.txt"  # Path to the extracted text file
output_json_file="data/output/results.json"  # Path to the output results file

# Step 1: Activate the first pyenv environment (Image-to-Text model)
echo "Activating the first pyenv environment (aivet)..."
pyenv shell aivet  # Activate the pyenv environment for bt1.py

echo "Running the first model (Image to Text extraction)..."
# Run bt1.py to extract text
python "runners/bt1.py" --input "$input_image" --output "$output_image_dir/extracted_text.txt"

# Check if the first model generated any output
if [ ! -f "$output_image_dir/extracted_text.txt" ]; then
  echo "Error: Image to Text model did not generate the expected output."
  exit 1
fi

# Step 2: Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"

# Step 3: Deactivate the first environment
pyenv shell --unset

# Step 4: Activate the second pyenv environment (Text Analysis model)
echo "Activating the second pyenv environment (aivet)..."
pyenv shell aivet  # Activate the pyenv environment for bt2.py

echo "Running the second model (Text analysis)..."
# Run bt2.py for text analysis and summarization
python "runners/bt2.py" --input "$output_image_dir/extracted_text.txt" --output "$output_json_file"

# Check if the second model generated the expected output
if [ ! -f "$output_json_file" ]; then
  echo "Error: Text Analysis model did not generate the expected output."
  exit 1
fi

# Step 5: Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()"

# Step 6: Deactivate the second environment
pyenv shell --unset

echo "Fast diagnosis completed."
