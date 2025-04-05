from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch


SECRET_KEY=config('OPENAI_API_KEY')


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
# Define prompt template
positive_feedback_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}")
    
])

negative_feedback_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative: {feedback}")
])

neutral_feedback_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response for more details for this neutral feedback: {feedback}.")
])

escalate_feedback_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}.")
])
    
# Define the feedback classification_template
classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}")
])

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x:"positive" in x,positive_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x:"negative" in x,negative_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x:"neutral" in x,positive_feedback_template | llm | StrOutputParser()
    ),
    escalate_feedback_template | llm | StrOutputParser()
)

# Create the classification chain

classification_chain = classification_template | llm | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Good_review = "The product is excellent. I really enjoyed using it and found it very helpful"
# Bad_review ="The product is terrible. It broke after just one use and the quality is very poor."
Neutral_review = "The product is okay. It works as expected but nothing exceptional."
# Default = "I'am not sure about the product yet. Can you tell me more about its features and benefits?"

# review = "The product is terrible. It broke after just one use and the quality is very poor."
# result = chain.invoke({"feedback":Neutral_review})

for c in chain.stream({"feedback":Neutral_review}):
    print(c,end="",flush=True)

# print(result) 