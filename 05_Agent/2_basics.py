from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, Tool, initialize_agent,tool
import datetime

SECRET_KEY = config('OPENAI_API_KEY')

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
)
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)

tools = [
    Tool(
        name="Get System Time",
        func=get_system_time,
        description="Use this to get the current system time. Input should be a time format string like '%Y-%m-%d %H:%M:%S'."
    )
]

# Use an agent type that allows LLM-based responses
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent="openai-functions",
    verbose=True
)

# Test queries
queries = [
    "What is the current time in South Africa? (You are in India) Just show current time not Date",
    "What is Django?"
]

for query in queries:
    result = agent_executor.invoke({"input": query})
    print(result)
