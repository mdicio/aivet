import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


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

    def extract_text(self, image_path: str, query: str) -> str:
        """
        Extracts structured text information from an image based on the provided query.

        :param image_path: The path to the image file containing the blood test results.
        :param query: The query prompt used to guide the extraction process.
        :return: The extracted text generated by the model.
        """
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs for the model in the chat format
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": self.query}],
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
            self.clear_cuda_memory()
            print(f"I2T Output: {output_text}")
            return output_text