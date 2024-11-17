class TextAnalyzer:
    def __init__(self, context, llm):
        """
        Initialize the TextAnalyzer with context, input question, and LLM.
        """
        self.context = context

        self.llm = llm  # The language model instance, e.g., OpenAI, HuggingFace, etc.

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

    def generate_prompt(self, mode):
        """
        Generate the appropriate prompt based on the selected mode.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt
        """
        if mode == "chain_of_thought":
            return self.cot_prompt.format(context=self.context)

        elif mode == "few_shot":
            return self.few_shot_prompt.format(context=self.context)

        elif mode == "general":
            return self.general_structure.format(
                context=self.context, input=self.input_question
            )

        else:
            return "Invalid mode."

    def analyze(self, mode):
        """
        Analyze the input question using the selected mode and return the generated prompt.
        :param mode: The mode to be used ('chain_of_thought', 'few_shot', or 'general')
        :return: The generated prompt for the analysis
        """
        prompt = self.generate_prompt(mode)
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
