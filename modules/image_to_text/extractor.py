import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import fitz  # PyMuPDF
import io
import os
import sys


class ImageToTextExtractor:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model_path = model_path
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.query = """Task: Extract and list all blood test details.

    Objective:
    1. Identify and extract **chemical names** from the blood test report.
    2. Capture the **detected levels** of each chemical.
    3. Extract the **normal range** for each corresponding chemical.

    Expected Output Format:
    - Chemical Name: [Name]
    - Detected Level: [Value]
    - Normal Range: [Min Value] - [Max Value]

    Please ensure accurate extraction, including any unit symbols (e.g., mg/dL, IU/L), and handle variations in formatting or alignment. Return results in a structured list format for easy readability."""

    def load_model(self):
        """
        Load the pre-trained model from the specified path.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            use_cache=True,
        )
        return model

    # Function to clear GPU memory
    def clear_cuda_memory(self):
        # Delete any previously loaded model (if applicable)
        del self.model  # Replace 'model' with the variable name of the first model
        torch.cuda.empty_cache()  # Clears the GPU cache

    def convert_pdf_to_images(self, pdf_path: str, zoom: float = 2.0) -> list:
        """
        Converts each page of a PDF to an in-memory image compatible with PIL.

        :param pdf_path: Path to the PDF file.
        :param zoom: Zoom factor for rendering the PDF pages.
        :return: List of PIL Image objects for each page.
        """
        images = []
        mat = fitz.Matrix(zoom, zoom)  # Zoom factor for better resolution
        try:
            doc = fitz.open(pdf_path)  # Open the PDF
            for page_num, page in enumerate(doc):  # Loop through pages
                pix = page.get_pixmap(matrix=mat)  # Render page as pixmap
                img_data = pix.tobytes("png")  # Convert pixmap to PNG byte data
                img = Image.open(io.BytesIO(img_data))  # Load image into PIL
                images.append(img.convert("RGB"))  # Convert to RGB
                print(f"Converted page {page_num + 1} to image.")
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            raise
        return images

    def extract_text(self, file_path: str, query: str) -> str:
        """
        Extracts structured text information from a PDF or image based on the provided query.

        :param file_path: The path to the PDF or image file containing the document.
        :param query: The query prompt used to guide the extraction process.
        :return: The concatenated text extracted from all pages of the document.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        extracted_text = ""

        if file_extension == ".pdf":
            print(f"Processing PDF file: {file_path}")
            images = self.convert_pdf_to_images(
                file_path
            )  # Convert PDF pages to images
            for page_num, image in enumerate(images):
                print(f"Extracting text from page {page_num + 1}...")
                extracted_text += self._process_image(image, query, page_num + 1)
                extracted_text += "\n\n"  # Separate text from different pages
        elif file_extension in [".png", ".jpeg", ".jpg"]:
            print(f"Processing image file: {file_path}")
            image = Image.open(file_path).convert("RGB")
            extracted_text = self._process_image(image, query)
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. Please provide a PDF or image file."
            )
        self.clear_cuda_memory()

        return extracted_text

    def _process_image(self, image: Image.Image, query: str, page_num=None) -> str:
        """
        Processes a single image and extracts text using the LLM model.

        :param image: PIL Image object to process.
        :param query: The query prompt to guide the extraction.
        :param page_num: Optional page number for logging.
        :return: The text extracted from the image.
        """
        # Prepare inputs for the model in the chat format
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_length": 2500,
            "do_sample": True,
            "top_k": 1,
            "temperature": 0.2,
        }

        with torch.no_grad():
            # Generate output from the model
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            output_text = self.tokenizer.decode(outputs[0])
            print(
                f"Page {page_num} Output: {output_text}"
                if page_num
                else f"Output: {output_text}"
            )
            return output_text
