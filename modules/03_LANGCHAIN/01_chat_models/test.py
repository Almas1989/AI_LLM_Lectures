from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
    num_predict=150
)

# user_message = "Где растут кактусы?"
# answer = llm.invoke(user_message)
# Для красивого вывода словаря
# print(repr(answer))

# Или доступ к отдельным полям:
# print(f"Content: {answer.content}")
# print(f"Model: {answer.response_metadata.get('model')}")
# print(f"Tokens: {answer.usage_metadata}")
# print(f"ID: {answer.id}")


messages = [
    SystemMessage(content="You translate Russian to English. Translate the user sentence and write only result:"),
    HumanMessage(content="Я создам успешный AI-продукт!")
]

ai_message = llm.invoke(messages)
print(ai_message.content)