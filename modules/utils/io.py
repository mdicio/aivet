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


# Function to append results to a JSON file
def append_results_to_json(output_file, result_data):
    """
    Appends a single result to a JSON file. Creates the file if it doesn't exist.
    """
    try:
        # Load existing data if the file exists
        with open(output_file, "r") as json_file:
            results = json.load(json_file)
    except FileNotFoundError:
        # Initialize as an empty list if the file doesn't exist
        results = []

    # Append the new result
    results.append(result_data)

    # Write the updated results back to the file
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
