import json
import time
import os
import sys
from pathlib import Path

# Импортируем ChatOpenAI, так как работаем через OpenRouter
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

# ==========================================
# 1. DYNAMIC CONFIGURATION LOADING
# ==========================================

# Construct the path to config.json relative to the script's location
# If config.json is in the same folder as this script:
def load_config(filename="config.json"):
    # Start at script location and traverse up parent directories
    current_path = Path(__file__).resolve().parent
    for parent in [current_path] + list(current_path.parents):
        check_path = parent / filename
        if check_path.exists():
            return check_path
    return None

config_file = load_config()

if not config_file:
    print("❌ Critical Error: config.json not found in any parent directory.")
    sys.exit(1)

# Load the file...
# ==========================================
# MISSING STEP: READ THE FILE
# ==========================================
try:
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"✅ Configuration loaded successfully from: {config_file}")
except json.JSONDecodeError:
    print(f"❌ Error: The file at {config_file} contains invalid JSON.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error loading config: {e}")
    sys.exit(1)

# ==========================================
# 2. CLIENT SETUP
# ==========================================
API_KEY = config["OPENROUTER_API_KEY"]
MODEL_NAME = config["MODEL_NAME"]

# ==========================================
# 2. ДАННЫЕ И ИНСТРУМЕНТ (Tool)
# ==========================================
ORDERS_STATUSES_DATA = {
    "a42": "Доставляется",
    "b61": "Выполнен",
    "k37": "Отменен",
}

# @tool - это "магия" LangChain. 
# Она сама создает JSON-схему для модели на основе типов Python (type hints) и описания (docstring).
@tool
def get_order_status(order_id: str = Field(description="Identifier of order")) -> str:
    """Get status of order by order identifier"""
    # Симуляция поиска в БД
    return ORDERS_STATUSES_DATA.get(order_id, f"Не существует заказа с order_id={order_id}")

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (OpenRouter)
# ==========================================
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model=MODEL_NAME,
    temperature=0,
    api_key=API_KEY,
    # Ключ берется автоматически из os.environ["OPENAI_API_KEY"]
)

# Связываем модель с инструментами
# LangChain сам преобразует функцию get_order_status в формат OpenAI Tools
llm_with_tools = llm.bind_tools([get_order_status])

# ==========================================
# 4. ВЫПОЛНЕНИЕ (Цепочка вызовов)
# ==========================================
print("--- Начало диалога ---")
messages = [
    HumanMessage(content="What about my order k37?")
]

# 1. Первый вызов LLM
# Модель должна понять, что нужно вызвать функцию
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print(f"AI решил: {ai_message.tool_calls}")

# 2. Обработка вызовов инструментов
if ai_message.tool_calls:
    for tool_call in ai_message.tool_calls:
        # Проверяем, какую функцию модель хочет вызвать
        if tool_call["name"] == "get_order_status":
            print(f"🔧 Выполняю инструмент: {tool_call['name']}")
            
            # В LangChain мы просто передаем tool_call в функцию.invoke
            # Она сама распакует аргументы и вернет правильный ToolMessage
            tool_message = get_order_status.invoke(tool_call)
            
            messages.append(tool_message)
            print(f"Результат инструмента: {tool_message.content}")

    # Небольшая пауза для реалистичности
    time.sleep(1)

    # 3. Финальный вызов LLM (чтобы он озвучил ответ пользователю)
    print("--- Генерация финального ответа ---")
    final_response = llm_with_tools.invoke(messages)
    messages.append(final_response)
    
    print(f"🤖 Bot: {final_response.content}")
else:
    print(f"🤖 Bot: {ai_message.content}")