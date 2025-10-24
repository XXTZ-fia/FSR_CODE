import json
from openai import OpenAI
import os
api_key = "sk-i9kLDp3SuH3xnNWEJkIeXvh8yx5F5XqZnakRQCrNr4MaIQu9"
base_url = "https://sg.uiuiapi.com/v1"


client = OpenAI(api_key=api_key, base_url=base_url)

with open("generate_question/prompt.json") as f:
    data = json.load(f)
    prompt = data["system"]
    output_format = data["output_format"]

with open('analysis_report.txt', "r", encoding="utf-8") as f:
    language_space=f.read()

output=None

for attempt in range(3):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(output_format, ensure_ascii=False, indent=2)},
                {"role": "user", "content": language_space}
            ]
        )
        output = response.choices[0].message.content
        break
    except Exception as e:
        print(f"The {attempt+1} attempt failed: {e}")

with  open ("generated_questions.json", "w") as f:
    f.write(output)