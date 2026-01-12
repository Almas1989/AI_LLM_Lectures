import json
import time
import os
from pathlib import Path
from openai import OpenAI  # Для OpenRouter используем этот клиент

# ==========================================
# 1. ЗАГРУЗКА КОНФИГУРАЦИИ (ПРЯМОЙ ПУТЬ)
# ==========================================

# ВАРИАНТ А: Если конфиг лежит В ТОЙ ЖЕ папке, что и этот скрипт:
# config_path = Path(__file__).parent / "config.json"

# ВАРИАНТ Б (Ваш случай): Если конфиг лежит в корне проекта (на 3-4 уровня выше).
# Чтобы не гадать с parents[3], укажите ПОЛНЫЙ ПУТЬ строкой (Nuclear Option):
# Обратите внимание на букву r перед строкой (raw string)
config_path = r"C:\D\обуч\stepik\LLM\Lectures\config.json"

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"✅ Конфигурация успешно загружена из: {config_path}")
except FileNotFoundError:
    print(f"❌ ОШИБКА: Файл не найден по пути: {config_path}")
    print("Проверьте, точно ли файл лежит именно там.")
    exit(1)

# Настройка клиента
API_KEY = config["OPENROUTER_API_KEY"]
# Используем бесплатную модель Mistral через OpenRouter
MODEL_NAME = config["MODEL_NAME"]

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ КЛИЕНТА (OpenRouter)
# ==========================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "FunctionCallingApp"
    }
)

# ==========================================
# 3. ЛОГИКА ФУНКЦИЙ (Ваш код)
# ==========================================
ORDERS_STATUSES_DATA = {
    "a42": "Доставляется",
    "b61": "Выполнен",
    "k37": "Отменен",
}

def get_order_status(order_id: str) -> str:
    return ORDERS_STATUSES_DATA.get(order_id, f"Не существует заказа с order_id={order_id}")

def cancel_order(order_id: str) -> str:
    if order_id not in ORDERS_STATUSES_DATA:
        return f"Не существует заказа с order_id={order_id}"
    if ORDERS_STATUSES_DATA[order_id] != "Отменен":
        ORDERS_STATUSES_DATA[order_id] = "Отменен"
        return "Заказ успешно отменен"
    return "Заказ уже отменен"

NAMES_TO_FUNCTIONS = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get status of order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order identifier"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel the order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order identifier"}
                },
                "required": ["order_id"],
            },
        },
    },
]

# ==========================================
# 4. ВЫПОЛНЕНИЕ ЗАПРОСА
# ==========================================
messages = [
    {
        "role": "user",
        "content": "Отмени заказ a42"
    },
]
print(f"User: {messages[0]['content']}")

# 1. Первый вызов (LLM решает, какой инструмент нужен)
chat_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
)

response_message = chat_response.choices[0].message
messages.append(response_message)

# Проверяем, есть ли вызов инструмента
if response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    print(f"Function Calling: {tool_call.function.name}")

    # 2. Выполнение функции Python
    function_name = tool_call.function.name
    function_params = json.loads(tool_call.function.arguments)
    function_result = NAMES_TO_FUNCTIONS[function_name](**function_params)
    
    print("function_name: ", function_name)
    print("function_params: ", function_params)
    print("function_result: ", function_result)

    # 3. Добавление результата в историю
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id, # Обязательно для OpenAI API
        "name": function_name,
        "content": str(function_result)
    })

    time.sleep(1)

    # 4. Финальный ответ LLM пользователю
    second_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    print("-" * 30)
    print("", second_response.choices[0].message.content)
    print("-" * 30)
else:
    print("Модель не вызвала функцию.")