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
animal_facts_template=ChatPromptTemplate([
    ("system", "You love facts and you tell facts about {animal} "),
    ("human", "Tell me {count} facts.")
    
])

# Define a prompt template for translate to French
translation_template=ChatPromptTemplate([
    ("system", "You are a translator and convert the provide text into {language}."),
    ("human", "Translate the following text to {language}:{text}")
])

def countWords(x):
    print(x)
    return f"Word count: {len(x.split())}\n{x}"

def set_data(output):
    return {"text":output,"language":"french"}

# Define additional processing steps using RunnableLambda
count_word = RunnableLambda(countWords)
prepare_for_translation = RunnableLambda(set_data)
#! prepare_for_translation = RunnableLambda(lambda output:{"text":output,"language":"french"})

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = animal_facts_template | llm | StrOutputParser()  | prepare_for_translation  | translation_template  | llm | StrOutputParser()


# run the chain

result = chain.invoke({"animal":"cat","count":2})

print(result)
