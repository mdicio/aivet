import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import argparse
from modules.image_to_text.extractor import ImageToTextExtractor


# Function to extract text from an image or PDF
def extract_text(file_path, model_path, output_path):
    # Initialize the extractor
    extractor = ImageToTextExtractor(model_path=model_path)
    file_extension = os.path.splitext(file_path)[1].lower()

    print(f"Received file: {file_path}")
    print(f"File extension: {file_extension}")

    # Check if file type is supported
    if file_extension in [".png", ".jpeg", ".jpg", ".pdf"]:
        # Extract text from the image
        extracted_text = extractor.extract_text(file_path, extractor.query)
        print(type(extracted_text))
        print("in bt1 extracted text", extracted_text)

        # Write the extracted text to the output file using UTF-8 encoding
        print("OUTPUT PATH EXTRACT TEXT", output_path)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Text extraction successful. Output saved to {output_path}")
        except Exception as e:
            print(f"Error writing text to file: {e}")
            sys.exit(1)

        return extracted_text
    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}. Please provide an image or PDF."
        )


# Main function to handle command-line arguments and extraction
def main():
    # Set up argument parsing
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
    input_file = args.input
    output_file = args.output

    # Model path (update this as needed)
    model_path = "nikravan/glm-4vq"  # Example model path

    try:
        # Call the function with the correct file path
        extracted_text = extract_text(input_file, model_path, output_file)
    except Exception as e:
        print(f"Error during text extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
