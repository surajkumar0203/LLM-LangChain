from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent,tool,initialize_agent
# from langchain_community.tools import DuckDuckGoSearchRun
import datetime
SECRET_KEY=config('OPENAI_API_KEY')



llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


# # default prompt for create_react_agent
prompt = hub.pull("hwchase17/react")
# # print(prompt)
tools = [
    get_system_time
]

agent = create_react_agent(
    llm,
    tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True
)


# # query = "Whats the weather of patna today?"
query = "What is the current time in South Africa? (You are in India) Just show current time not Date "
# query = "Who is Bhagat Singh"
result=agent_executor.invoke({"input": query})

print(result)