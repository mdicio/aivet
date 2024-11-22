import json
import time
import warnings
from modules.text_analysis.analyzer import TextAnalyzer
from modules.utils.io import read_from_text_file, write_to_text_file
from modules.utils.llm_helpers import check_gpu_memory

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def test_generate_response2(input_file, model_path, llm_conf=None, output_file=None):
    """
    Generates response2 (summary) from a given response1 text file.

    Args:
        input_file (str): Path to the file containing response1 text.
        model_path (str): Path to the LLM model to use.
        llm_conf (dict): Configuration for the LLM model.
        output_file (str): Path to save the generated response2. Optional.
    """
    # Load response1 from the provided file
    response1 = read_from_text_file(input_file)
    print(f"Loaded response1 from: {input_file}")

    # Set default LLM configuration if not provided
    if llm_conf is None:
        llm_conf = {
            "temperature": 0.2,
            "repetition_penalty": 1.11,
            "top_p": 0.95,
            "n_gpu_layers": 87,
        }

    # Initialize the TextAnalyzer with dummy context (response1 is used directly)
    analyzer = TextAnalyzer(model_path, llm_conf)

    # Check GPU memory before starting
    check_gpu_memory()

    # Generate response2 (summary)
    start_time = time.time()
    response2 = analyzer.summarize(response1)
    elapsed_time = round(time.time() - start_time, 4)

    # Check GPU memory after processing
    check_gpu_memory()

    # Display the summary
    print("Generated response2 (summary):")
    print(response2)

    # Optionally, save the summary to a file
    if output_file:
        write_to_text_file(response2, output_file)
        print(f"Response2 saved to: {output_file}")

    # Return the summary and processing time
    return response2, elapsed_time


if __name__ == "__main__":
    # Example test parameters
    input_file = "data/output/t2t_output_CalmeRys-78B-Orpo-v01i1-IQ3_Sgguf_chain_of_thought_4422.2193_t0.88_r1.12_p0.93.txt"  # Path to response1 file
    model_path = "models/Llama-3.2-3B-Instruct-Q6_K_L.gguf"  # Model path
    output_file = "data/output/response2_summary.txt"  # Output file path (optional)

    # Define a sample LLM configuration
    llm_conf = {
        "temperature": 0.2,
        "repetition_penalty": 1.11,
        "top_p": 0.95,
        "n_gpu_layers": 87,
    }

    # Run the test
    response2, time_taken = test_generate_response2(
        input_file, model_path, llm_conf, output_file
    )
    print(f"Summary generation completed in {time_taken} seconds.")
