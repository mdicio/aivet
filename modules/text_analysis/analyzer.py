from llama_cpp import Llama
import torch


class TextAnalyzer:
    def __init__(self, model_path, llm_conf):
        """
        Initialize the TextAnalyzer with context, input question, and LLM.
        """
        self.llm_conf = llm_conf
        self.model_path = model_path
        self.load_llm(
            self.llm_conf, self.model_path
        )  # The language model instance, e.g., OpenAI, HuggingFace, etc.

        # Define the general prompt structure as a template
        self.general_structure = """
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context in the section demarcated by "```" to answer the question.
        The context may contain multiple question answer pairs as an example, just answer the final question provided after the context.
        If you don't know the answer just say that you don't know. Use three sentences maximum and keep the answer concise.

        {context}
        Question: {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        # Define the specific question as a string
        self.input_question = """
        1. For each tested molecule check the detected level.
        2. For each tested molecule analyze the normal range.
        3. For each tested molecule infer whether the detected level is outside the normal range.
        4. Select and return exclusively the molecules that fall outside the normal range, citing both the etected level and the range.
        5. Provide a comprehensive health diagnosis for the dog based on detected abnormalities.
        """

        self.cot_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        You are a veterinary assistant analyzing blood test results for a dog. Follow these steps to provide your analysis:

        Step 1: Review the list of blood test results.
        Step 2: Identify any molecules that are outside the normal range. For each molecule, follow this reasoning process:
        - State the molecule's name.
        - Compare the detected level with the normal range.
        - Explain why this molecule is considered "High" or "Low" based on the comparison.
        - Provide the detected level and the normal range.
        Step 3: Summarize what the identified abnormalities might indicate about the dog's health, based on veterinary guidelines and typical implications of abnormal levels.
        Step 4: If all molecules are within the normal range, just say "All results are normal."

        **Example Analysis:**

        Blood test results:
        - AST: Detected level 50, Normal range 10-45
        - ALT: Detected level 25, Normal range 10-60
        - ALP: Detected level 200, Normal range 45-152
        - Glucose: Detected level 65, Normal range 70-110

        Analysis:
        1. **AST**: The detected level is 50, which is above the normal range of 10-45. This indicates a "High" reading. Elevated AST can be a sign of liver or muscle damage, as the enzyme is released into the bloodstream when these tissues are damaged.
        2. **ALP**: The detected level is 200, which is higher than the normal range of 45-152. This is considered "High" and can indicate liver disease, bone disorders, or certain endocrine disorders.
        3. **Glucose**: The detected level is 65, which is below the normal range of 70-110. This indicates a "Low" reading. Low glucose levels can suggest hypoglycemia, which may be due to issues like insulin overdose, liver disease, or severe infections.

        **Summary**: Elevated AST and ALP levels may point to potential liver issues, while low glucose could suggest hypoglycemia. Further diagnostic tests are recommended to confirm the underlying condition.

        ---

        Now, analyze the following blood test results:

        Blood test results:
        {context}

        Please follow the reasoning steps to provide your analysis.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        # Define the few-shot examples
        self.few_shot_prompt = """You are a veterinary assistant reviewing blood test results for a dog. Your task is to identify only the molecules that are outside the normal range. For each abnormal molecule, indicate whether it is "High" or "Low", along with the detected level and the normal range. After listing the abnormal molecules, provide a brief summary of what the abnormalities might indicate about the dog's health.
        Use these examples to understand the structure of the input and the expected output you are required to produce.

        Example 1:
        Blood test results:
        - AST: Detected level 50, Normal range 10-45
        - ALT: Detected level 25, Normal range 10-60
        - ALP: Detected level 200, Normal range 45-152
        - Glucose: Detected level 65, Normal range 70-110

        Analysis:
        - AST: High (Detected level 50, normal range 10-45)
        - ALP: High (Detected level 200, normal range 45-152)
        - Glucose: Low (Detected level 65, normal range 70-110)

        Summary: Elevated AST and ALP may indicate liver issues, while low glucose could suggest hypoglycemia.

        Example 2:
        Blood test results:
        - AST: Detected level 35, Normal range 10-45
        - ALT: Detected level 20, Normal range 10-60
        - Glucose: Detected level 80, Normal range 70-110

        Analysis:
        All results are normal.

        Now, please analyze the following blood test results:
        ### New Input:
        {context}
        """

        self.summary_prompt = """You are a health analysis assistant tasked with interpreting blood test results. Focus on identifying **abnormal readings** in the provided data and give a detailed explanation of the potential causes and health implications for these abnormalities. Provide context-specific details about what the abnormal readings might indicate about the dog's health, including **possible conditions** or **diseases** they could be associated with.

        Only recommend **specific next steps** if the abnormalities clearly suggest that further medical tests or actions are required. These steps could include re-checking the results, scheduling specific diagnostic exams, or addressing underlying health concerns. Avoid general guidelines unless absolutely necessary.

        List and explain the **specific molecules** with abnormal readings, and only recommend next steps if the abnormalities suggest the need for additional investigations.

        Answer the following question in **no more than 5 sentences**:

        **Main Question**: Based on the blood test results, what are the key health implications of the abnormalities, and what specific next steps should be taken, if any?

        Context:
        {context}
        """

    def unload_llm(self):
        del self.llm
        torch.cuda.empty_cache()

    def load_llm(self, llm_conf, model_path):

        self.llm = Llama(
            model_path=model_path,  # Download the model file first
            n_ctx=5000,  # The max sequence length to use
            n_threads=12,  # CPU threads
            n_gpu_layers=llm_conf["n_gpu_layers"],  # Number of layers to offload to GPU
            temperature=llm_conf["temperature"],
            repetition_penalty=llm_conf["repetition_penalty"],
            top_p=llm_conf["top_p"],
        )

    def generate_analyzer_prompt(self, mode, context):
        """
        Generate the appropriate prompt based on the selected mode.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt
        """
        if mode == "chain_of_thought":
            return self.cot_prompt.format(context=context)

        elif mode == "few_shot":
            return self.few_shot_prompt.format(context=context)

        elif mode == "general":
            return self.general_structure.format(
                context=context, input=self.input_question
            )

        else:
            return "Invalid mode."

    def generate_summarizer_prompt(self, context):
        """
        Generate the appropriate prompt based on the selected mode.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt
        """
        return self.summary_prompt.format(context=context)

    def analyze(self, mode, context):
        """
        Analyze the input question using the selected mode and return the generated prompt.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt for the analysis
        """
        prompt = self.generate_analyzer_prompt(mode, context)
        # Now call inference with the generated prompt
        result = self.llm_inference(prompt)
        return result

    def summarize(self, context):
        """
        Analyze the input question using the selected mode and return the generated prompt.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt for the analysis
        """
        prompt = self.generate_summarizer_prompt(context)
        # Now call inference with the generated prompt
        result = self.llm_inference(prompt)
        return result

    def llm_inference(self, prompt):
        """
        Perform inference using the language model with the generated prompt.
        :param prompt: The generated prompt string.
        :return: The model's generated response.
        """
        output = self.llm(
            prompt,
            max_tokens=None,  # Generate up to 2048 tokens, can adjust as needed
            echo=True,
        )

        # Extract the generated text (removing the initial prompt part)
        result = output["choices"][0]["text"][len(prompt) :]
        return result
