import json
from pathlib import Path
from openai import OpenAI

# Определяем путь к config.json
current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parents[2]
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=config["OPENROUTER_API_KEY"],
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="meta-llama/llama-3.3-70b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)