import json
from pathlib import Path
from openai import OpenAI

# Определяем путь к config.json
current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parents[2]
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

API_KEY = config["OPENROUTER_API_KEY"]
BASE_URL = "https://api.mistral.ai/v1"
MODEL_NAME = "mistral-small-latest"


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

message = "Привет! Когда ждать появления AGI?"

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Ты чокнутый профессор, который думает, что AGI сидит у него в подвале",
        },
        {
            "role": "user",
            "content": message,
        }
    ],
    model=MODEL_NAME,
    temperature=0.1
)

print(chat_completion.choices[0].message.content)
