import openai
import json
from api_agent import api_agent
from local_agent import local_agent

def init_agent(agent_type, model_name):

    if agent_type=="api":
        with open("api_message.json", "r") as f:
            config = json.load(f)
            api_key = config.get("api_key")
            base_url = config.get("base_url")
        test_agent=api_agent(api_key=api_key, base_url=base_url, model_name=model_name)

    elif agent_type=="local":
        test_agent=local_agent(model_name=model_name)

    return test_agent

def ask_question(test_agent):
    attempt = 0
    for i in range(3):
        try:
            output = test_agent.ask_question()
            return output
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
