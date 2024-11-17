import torch
from llama_cpp import Llama
from modules.text_analysis.analyzer import TextAnalyzer
import json
import time
import warnings
import numpy as np
import random

warnings.filterwarnings("ignore", category=FutureWarning)


def check_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
    print(f"Memory Allocated: {allocated_memory} GB")
    print(f"Memory Reserved: {reserved_memory} GB")


def sample_llm_conf():
    # Define the sampling ranges for each parameter
    temp_range = (0.0, 1.0)  # temp typically between 0 and 1
    rp_range = (1.0, 1.2)  # rp could range between 0 and 3 (just as an example)
    topp_range = (0.9, 0.97)  # topp typically between 0 and 1

    # Randomly sample values within these ranges
    llm_conf = {
        "temp": round(random.uniform(*temp_range), 2),
        "rp": round(random.uniform(*rp_range), 2),
        "topp": round(random.uniform(*topp_range), 2),
    }

    return llm_conf


def analyze_text(
    context,
    model_path,
    mode="general",
    llm_conf={"temp": 0.2, "rp": 1.11, "topp": 0.95},
):

    llm = Llama(
        model_path=model_path,  # Download the model file first
        n_ctx=5000,  # The max sequence length to use
        n_threads=12,  # CPU threads
        n_gpu_layers=87,  # Number of layers to offload to GPU
        temperature=llm_conf["temp"],
        repetition_penalty=llm_conf["rp"],
        top_p=llm_conf["topp"],
    )

    analyzer = TextAnalyzer(context, llm)
    response = analyzer.analyze(
        mode=mode,
    )  # Example using general prompt mode
    check_gpu_memory()

    # Clear GPU memory after use
    del llm
    torch.cuda.empty_cache()
    check_gpu_memory()

    return response


def read_from_text_file(filename="output.txt"):
    """
    Reads the extracted text from a plain text file.

    :param filename: The name of the text file to read.
    :return: The content of the text file as a string.
    """
    try:
        # Open the file in read mode and ensure UTF-8 encoding
        with open(filename, "r", encoding="utf-8") as file:
            extracted_text = file.read()  # Read the entire content of the file
        print(f"Extracted text has been read from {filename}")
        return extracted_text
    except Exception as e:
        print(f"Error reading from file: {e}")
        return None


def write_to_text_file(extracted_text, filename="output.txt"):
    """
    Writes the extracted text to a plain text file.

    :param extracted_text: The text to be written to the file.
    :param filename: The name of the text file where the content will be saved.
    """
    try:
        # Open the file in write mode and ensure UTF-8 encoding
        with open(filename, "w", encoding="utf-8") as file:
            file.write(extracted_text)
        print(f"Extracted text has been written to {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    input_path = (
        "data/processed/extracted_text.txt"  # Path to the JSON file with extracted text
    )

    t2t_llm = "models/Llama-3.2-3B-Instruct-Q6_K_L.gguf"
    # Define your list of modes
    modes = ["chain_of_thought", "general", "few_shot"]
    # Choose a random mode from the list
    mode = random.choice(modes)
    print("PROMPT MODE", mode)

    check_gpu_memory()
    t0 = time.time()
    llm_conf = sample_llm_conf()
    # llm_conf = {"temp": 0.2, "rp": 1.11, "topp": 0.95}
    temperature = llm_conf["temp"]
    repetition_penalty = llm_conf["rp"]
    top_p = llm_conf["topp"]
    print(llm_conf)
    extracted_text = read_from_text_file(input_path)
    response = analyze_text(extracted_text, t2t_llm, mode, llm_conf)

    tid = round(time.time() - t0, 4)
    mname = t2t_llm.strip("models/").replace(".", "")
    write_to_text_file(
        response,
        filename=f"data/output/t2t_output_{mname}_{mode}_{tid}_t{temperature}_r{repetition_penalty}_p{top_p}.txt",
    )
    print(response)
