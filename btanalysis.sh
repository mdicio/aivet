#!/bin/bash

# Define the number of iterations
num_iterations=100

for ((i=1; i<=num_iterations; i++))
do
  echo "Iteration $i of $num_iterations"

  # Step 4: Activate the second pyenv environment
  echo "Activating the second pyenv environment..."
  pyenv activate llama32  # Replace with the name of your pyenv environment

  # Run the Python script
  echo "Running the second model (Text analysis)..."
  python bt2.py  # No need to pass mode since it's randomized within the script

  # Step 5: Clear GPU memory
  echo "Clearing GPU memory..."
  python -c "import torch; torch.cuda.empty_cache()"

  # Step 6: Deactivate the pyenv environment
  pyenv deactivate

  echo "Iteration $i completed."
  echo "-------------------------------"

  # Optional: Add a short delay between runs if needed
  sleep 1  # Uncomment to add a delay between iterations
done

echo "All iterations completed successfully."
