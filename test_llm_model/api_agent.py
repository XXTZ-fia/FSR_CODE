import openai

class api_agent:
    def __init__(self, api_key, base_url, model_name):
        self.api_key=api_key
        self.base_url=base_url
        self.model_name=model_name
        self.prompt=[]

    def increase_input(self, questions):
        self.prompt.append({"role": "user", "content": questions})

    def ask_question(self):
        Client=openai.Client(api_key=self.api_key,base_url=self.base_url)
        try:
            print(self.prompt)
            response=Client.chat.completions.create(
                model=self.model_name,
                messages=self.prompt
            )
            output = response.choices[0].message.content
        except Exception as e:
            print("Error parsing response:", e)
            output=""
        return output