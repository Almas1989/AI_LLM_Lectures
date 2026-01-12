import json
import os
from pathlib import Path
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # Используем клиент OpenAI для OpenRouter

# --- 1. Конфигурация ---
# Определяем путь к config.json
current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parents[2]
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

os.environ["OPENAI_API_KEY"] = config["OPENROUTER_API_KEY"] 

DEFAULT_SESSION_ID = "default"
chat_history = InMemoryChatMessageHistory()

# --- 2. Промпт ---
messages = [
    ("system", "You are an expert in {domain}. Your task is answer the question as short as possible"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
]
prompt = ChatPromptTemplate(messages)

# --- 3. Триммер (Управление контекстным окном) ---
# token_counter=len считает количество сообщений, а не токенов слов.
# max_tokens=10 означает "хранить последние 10 сообщений".
trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=10,
    start_on="human",
    end_on="human",
    include_system=True,
    allow_partial=False
)

# --- 4. Инициализация модели (Devstral через OpenRouter) ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/devstral-2512:free",
    temperature=0,
)

# --- 5. Сборка цепочки (LCEL) ---
# Поток: (Input) -> Prompt -> Trimmer (обрезает лишнее) -> LLM
chain = prompt | trimmer | llm

# Добавляем управление историей
chain_with_history = RunnableWithMessageHistory(
    chain, 
    lambda session_id: chat_history,
    input_messages_key="question", 
    history_messages_key="history"
)

# Добавляем парсер строки в конце, чтобы получать чистый текст
final_chain = chain_with_history | StrOutputParser()

# --- 6. Запуск цикла чата ---
if __name__ == "__main__":
    try:
        domain = input('Choice domain area: ')
        print(f"--- Chat initialized for domain: {domain} ---")
        print("(Press Ctrl+C to exit)\n")

        while True:
            user_question = input('You: ')
            if not user_question: break # Защита от пустого ввода
            
            print('Bot: ', end="", flush=True)
            
            # Потоковая передача ответа (Streaming)
            for answer_chunk in final_chain.stream(
                    {"domain": domain, "question": user_question},
                    config={"configurable": {"session_id": DEFAULT_SESSION_ID}},
            ):
                print(answer_chunk, end="", flush=True)
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nExiting chat. Goodbye!")