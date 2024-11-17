import os
from modules.image_to_text.extractor import ImageToTextExtractor
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def check_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
    print(f"Memory Allocated: {allocated_memory} GB")
    print(f"Memory Reserved: {reserved_memory} GB")


def extract_text(file_path, model_path):
    extractor = ImageToTextExtractor(model_path=model_path)
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in [".png", ".jpeg", ".jpg", ".pdf"]:
        return extractor.extract_text(file_path, extractor.query)
    else:
        raise ValueError("Unsupported file type. Please provide an image or PDF.")


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
    file_path = "data/input/202410_hope_bio.png"
    i2t_llm = "nikravan/glm-4vq"
    output_path = "data/processed/extracted_text.txt"  # Save output to this file

    check_gpu_memory()
    extracted_text = extract_text(file_path, i2t_llm)
    write_to_text_file(extracted_text, output_path)

    torch.cuda.empty_cache()
    check_gpu_memory()
