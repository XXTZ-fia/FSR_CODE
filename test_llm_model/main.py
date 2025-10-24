import os
import json
from agent import init_agent
from api_agent import api_agent
from local_agent import local_agent

model_name=input("Please enter the model name: ")
agent_type=input("Please enter the agent type (api/local): ")

#如果要清除记忆要重新初始化agent
test_agent = init_agent(agent_type, model_name)

answer=[]
with open("test_llm_model/questions.json", "r") as f:
    data = json.load(f)
    count=1
    for item in data["questions"]:
        question = item["question"]
        expected_answer = item["expected_answer"]
        test_agent.increase_input(question)
        response = test_agent.ask_question()
        print("Response:", response)
        answer.append({
            f"question{count}": response
        })
        count += 1

print("Final Answers:", answer)