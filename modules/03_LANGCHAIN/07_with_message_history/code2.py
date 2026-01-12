import json
import os
import time
from pathlib import Path

# Заменили импорт Mistral на OpenAI (для работы через OpenRouter)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Настройка ключа
# Определяем путь к config.json
current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parents[2]
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

os.environ["OPENAI_API_KEY"] = config["OPENROUTER_API_KEY"] 

DEFAULT_SESSION_ID = "default"
chat_history = InMemoryChatMessageHistory()

# 2. Инициализация модели через OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/devstral-2512:free", 
    temperature=0,
)

# 3. Настройка триммера (обрезка истории)
# Внимание: token_counter=len считает количество сообщений (если передается список), 
# поэтому max_tokens=6 здесь означает "оставить последние 6 сообщений".
trimmer = trim_messages(
    strategy="last",
    token_counter=len, # Используем len как простую функцию подсчета
    max_tokens=6,      # Храним последние 6 сообщений
    start_on="human",
    end_on="human",
    include_system=True,
    allow_partial=False
)

# 4. Сборка цепочки
chain = trimmer | llm
chain_with_history = RunnableWithMessageHistory(chain, lambda session_id: chat_history)

# 5. Первый вызов (Знакомство)
print("--- Сообщение 1: Знакомство ---")
chain_with_history.invoke(
    [HumanMessage("Hi, my name is Bob!")],
    config={"configurable": {"session_id": DEFAULT_SESSION_ID}},
)

# Имитация задержки (как в вашем примере)
time.sleep(2)

# 6. Второй вызов (Проверка памяти)
print("--- Сообщение 2: Вопрос ---")
ai_message = chain_with_history.invoke(
    [HumanMessage("What is my name?")],
    config={"configurable": {"session_id": DEFAULT_SESSION_ID}},
)

print(f"Ответ AI: {ai_message.content}")