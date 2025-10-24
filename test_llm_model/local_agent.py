import json
import requests
import os

class local_agent:
    def __init__(self, model_name, base_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.base_url = base_url
        self.prompt = []

    def increase_input(self, questions):
        self.prompt.append({"role": "user", "content": questions})

    def ask_question(self):
        payload = {
            "model": self.model_name,
            "messages": self.prompt
        }

        try:
            response = requests.post(self.base_url, json=payload)
            data = response.json()
            output = data.get("message", {}).get("content", "")
        except Exception as e:
            print("Error parsing response:", e)
            output = ""
        return output