from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel


SECRET_KEY=config('OPENAI_API_KEY')


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
# Define prompt template
summary_template=ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic."),
    ("human", "Provide a brief summary of the movie {movie_name}.")
    
])

def analyze_plot(plot):
    # print("plot : ",plot)
    plot_template=ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {plot}. What are their strengths and weaknesses?")
    ])
    return plot_template.format_prompt(plot=plot)

def analyze_characters(character):
    # print("character : ",character)
    character_template=ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human", "Analyze the plot: {character}. What are its strengths and weaknesses?")
    ])
    return character_template.format_prompt(character=character)

# Simplify branches with LCEL
plot_branch_chain = RunnableLambda(lambda x:analyze_plot(x)) | llm | StrOutputParser()



characters_branch_chain = RunnableLambda(lambda x:analyze_characters(x)) | llm | StrOutputParser()

def combine_verdicts(x):
    return f"Description: \n{x["branch"]['output']}\n\n plot:\n{x["branch"]["plot"]}\n\n character:\n {x["branch"]["character"]}"


# Create the combined chain using LangChain Expression Language 

chain = (
   summary_template | 
   llm | 
   StrOutputParser() |
   RunnableParallel(branch={"output":RunnableLambda(lambda x:x),"plot":plot_branch_chain,"character":characters_branch_chain}) |
   RunnableLambda(lambda x:combine_verdicts(x))
)

# result = chain.invoke({"movie_name":"Dil To Pagal hai"})
for c in chain.stream({"movie_name":"Dil To Pagal hai"}):
    print(c,end="",flush=True)
# print(result)