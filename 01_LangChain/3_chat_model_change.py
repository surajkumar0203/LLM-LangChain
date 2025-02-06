from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint
SECRET_KEY=config('OPENAI_API_KEY')

# repo_id = "deepseek-ai/DeepSeek-R1"


# llm = HuggingFaceEndpoint(
#     repo_id="gpt-3.5-turbo",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=config('huggingfacehub_api_token'),
# )
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)

messages = [
    
        # SystemMessage(content="You are a helpful assistant"),
   
   
        HumanMessage(content="Who developed you?"),
        AIMessage(content="I was developed through a collaboration between Mistral AI and NVIDIA."),
        HumanMessage(content="Who are You? tell me about your self"),
    
]


res=llm.invoke(messages)

print(res)

