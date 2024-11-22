import json
import sys
import warnings
import sys
import os
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


from modules.text_analysis.analyzer import TextAnalyzer
from modules.utils.llm_helpers import check_gpu_memory

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CONFIG = {
    "model_path": "models/Llama-3.2-3B-Instruct-Q6_K_L.gguf",  # Example model path
    "default_llm_conf": {
        "temperature": 0.2,
        "repetition_penalty": 1.11,
        "top_p": 0.95,
        "n_gpu_layers": 87,
    },
}


def analyze_text(input_file_path, output_file_path, model_path):

    # Read the extracted text from the file
    with open(input_file_path, "r", encoding="utf-8") as f:
        context = f.read()

    # Analyze the text with the LLM model
    analyzer = TextAnalyzer(model_path, DEFAULT_CONFIG["default_llm_conf"])
    check_gpu_memory()

    response1 = analyzer.analyze(mode="chain_of_thought", context=context)
    check_gpu_memory()

    response2 = analyzer.summarize(response1)
    check_gpu_memory()

    # response1 = "PIERO é UN NABBO"
    # response2 = "BEBO LA TROLLA"
    # Write the analysis and summary to the output JSON file
    output_data = {"analysis": response1, "summary": response2}
    with open(output_file_path, "w") as f:
        json.dump(output_data, f)

    return response1, response2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from an image or PDF file."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input image or PDF."
    )
    parser.add_argument("--output", required=True, help="Path to the output text file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the input and output file paths
    input_text_file = args.input
    output_file = args.output

    # Model path can be passed or set to default
    model_path = "models/Llama-3.2-3B-Instruct-Q6_K_L.gguf"
    try:
        analyze_text(input_text_file, output_file, model_path)
        print(f"Text analysis completed, saved to {output_file}")
    except Exception as e:
        print(f"Error during text analysis: {e}")
        sys.exit(1)
