from langchain_openai import ChatOpenAI
from decouple import config

SECRET_KEY=config('OPENAI_API_KEY')

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)

result = llm.invoke("Who is the current prime minister of India?")

print(result)