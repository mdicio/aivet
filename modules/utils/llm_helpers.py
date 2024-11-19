import random
import torch


def sample_llm_conf():
    # Define the sampling ranges for each parameter
    temp_range = (0.0, 1.0)  # temp typically between 0 and 1
    rp_range = (1.0, 1.2)  # rp could range between 0 and 3 (just as an example)
    topp_range = (0.9, 0.97)  # topp typically between 0 and 1

    # Randomly sample values within these ranges
    llm_conf = {
        "temperature": round(random.uniform(*temp_range), 2),
        "repetition_penalty": round(random.uniform(*rp_range), 2),
        "top_p": round(random.uniform(*topp_range), 2),
    }

    return llm_conf


def check_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
    print(f"Memory Allocated: {allocated_memory} GB")
    print(f"Memory Reserved: {reserved_memory} GB")
