from read_paper.deepseek import ds_client

class LLM:
    def __init__(self):
        # self.model_name = model_name
        # self.temperature = temperature
        # self.max_tokens = max_tokens
        self.client = ds_client
    def generate_response(self, prompt):
        # Code to generate response using the specified LLM model
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    llm = LLM()
    response = llm.generate_response("你是谁？")
    print(response)