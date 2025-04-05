from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.schema.output_parser import StrOutputParser

SECRET_KEY=config('OPENAI_API_KEY')


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
# Define prompt template
prompt_template=ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
    
])

# Create a combine chain using LangChain Express
# | pipe is to connect another different task (using for chain)
chain = prompt_template | llm | StrOutputParser()


# StrOutputParser =>  extrate content
result=chain.invoke({'topic':'cat'})
print(result)

