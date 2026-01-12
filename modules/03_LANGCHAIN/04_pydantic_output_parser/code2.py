import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
# Определяем путь к config.json
current_script_path = Path(__file__).resolve().parent
project_root = current_script_path.parents[2]
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

os.environ["OPENAI_API_KEY"] = config["OPENROUTER_API_KEY"] 

# Точный ID модели из вашего JS-примера
MODEL_ID = "mistralai/devstral-2512:free"

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
# ==========================================
# Мы используем ChatOpenAI, переопределяя base_url на OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model=MODEL_ID,
    temperature=0,
    # OpenRouter рекомендует добавлять эти заголовки
    default_headers={
        "HTTP-Referer": "http://localhost:8000", # URL вашего приложения (можно любой для тестов)
        "X-Title": "MyParsingApp"               # Название вашего приложения
    }
)

# ==========================================
# 3. НАСТРОЙКА ПАРСЕРА (Ваша логика)
# ==========================================
class Person(BaseModel):
    firstname: str = Field(description="Имя персоны")
    lastname: str = Field(description="Фамилия персоны")
    age: int = Field(description="Возраст персоны (число)")

parser = PydanticOutputParser(pydantic_object=Person)

# Для экспериментальных моделей (Devstral) лучше давать очень строгий промпт
messages = [
    (
        "system",
        "You are a helpful assistant that extracts data.\n"
        "Your ONLY task is to return a JSON object based on the user query.\n"
        "Do NOT include markdown formatting (like ```json ... ```).\n"
        "Do NOT output any conversational text.\n"
        "{format_instructions}"
    ),
    ("human", "{user_query}")
]

prompt_template = ChatPromptTemplate(messages)

prompt_value = prompt_template.invoke(
    {
        "format_instructions": parser.get_format_instructions(),
        "user_query": "Генрих Смит был восемнадцатилетним юношей, мечтающим уехать в город"
    }
)

# ==========================================
# 4. ЗАПУСК
# ==========================================
try:
    print(f"📡 Отправка запроса к {MODEL_ID}...")
    
    # Получаем ответ от модели
    answer = llm.invoke(prompt_value.to_messages())
    
    # Для отладки: выводим "сырой" ответ модели
    print(f"📝 Сырой ответ модели:\n{answer.content}\n")
    
    # Парсим в объект
    parsed_obj = parser.invoke(answer)
    
    print("✅ Успешно распарсено Pydantic:")
    print(f"--> Имя: {parsed_obj.firstname}")
    print(f"--> Фамилия: {parsed_obj.lastname}")
    print(f"--> Возраст: {parsed_obj.age}")

except Exception as e:
    print(f"❌ Ошибка: {e}")