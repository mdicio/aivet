import torch
import json
import time
import warnings
import random
from modules.text_analysis.analyzer import TextAnalyzer
from modules.utils.io import read_from_text_file, append_results_to_json
from modules.utils.llm_helpers import check_gpu_memory, sample_llm_conf

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration for the script
DEFAULT_CONFIG = {
    "model_path": "models/Llama-3.2-3B-Instruct-Q6_K_L.gguf",  # Default model path
    "text_input_path": "data/processed/extracted_text.txt",  # Path to input text file
    "output_file": "data/output/results.json",  # Main JSON output file
    "default_modes": [
        "chain_of_thought",
        "general",
        "few_shot",
    ],  # Available prompt modes
    "default_llm_conf": {
        "temperature": 0.2,
        "repetition_penalty": 1.11,
        "top_p": 0.95,
        "n_gpu_layers": 21,
    },
}


# Function to analyze text using a specified model and configuration
def analyze_text(context, model_path, mode="general", llm_conf=None):
    """
    Analyzes the given text context with a specified model and mode.
    """
    if llm_conf is None:
        llm_conf = DEFAULT_CONFIG["default_llm_conf"]

    analyzer = TextAnalyzer(context, model_path, llm_conf)
    check_gpu_memory()

    response1 = analyzer.analyze(mode=mode, context=context)
    check_gpu_memory()

    response2 = analyzer.summarize(response1)
    check_gpu_memory()

    return response1, response2


# Main script execution
if __name__ == "__main__":
    # Load configuration
    config = DEFAULT_CONFIG
    model_path = config["model_path"]
    text_input_path = config["text_input_path"]
    output_file = config["output_file"]
    modes = config["default_modes"]

    # Randomly select a mode and sample model configurations
    mode = random.choice(modes)
    llm_conf = sample_llm_conf()
    print(f"Selected Mode: {mode}")
    print(f"LLM Configuration: {llm_conf}")

    # Read input context
    context = read_from_text_file(text_input_path)

    # Record start time
    start_time = time.time()

    # Perform analysis
    response1, response2 = analyze_text(context, model_path, mode, llm_conf)

    # Calculate processing time
    elapsed_time = round(time.time() - start_time, 4)

    # Create structured output data
    result_data = {
        "model": model_path,
        "mode": mode,
        "context": config["text_input_path"],
        "llm_configuration": llm_conf,
        "responses": {
            "analysis": response1,
            "summary": response2,
        },
        "processing_time_seconds": elapsed_time,
    }

    # Append results to the main JSON file
    append_results_to_json(output_file, result_data)

    print(f"Results appended to: {output_file}")
    print(f"Processing completed in {elapsed_time} seconds.")
