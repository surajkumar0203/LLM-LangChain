from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableSequence


SECRET_KEY=config('OPENAI_API_KEY')


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
# Define prompt template
prompt_template=ChatPromptTemplate([
    ("system", "You love facts and you tell facts about {animal} "),
    ("human", "Tell me {count} facts.")
    
])

def call_LLM(x):
    # print(x)
    # print(x.messages)
    return llm.invoke(x.messages)

def prompt(x):
    print(x)
    return prompt_template.format_prompt(**x)




# Create individual runnables (steps in the chain)

# behind the sean
# x={"animal":"cat","count":2}
format_prompt = RunnableLambda(prompt)
# format_prompt = RunnableLambda(lambda x:prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(call_LLM)
# invoke_model = RunnableLambda(lambda x:llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x:x.content)






# Create the RunnableSequence (equivalent to the LCEL chain)
# chain = format_prompt | invoke_model | parse_output
chain = RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)

print(chain.invoke({"animal":"cat","count":2}))