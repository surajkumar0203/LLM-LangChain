from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
SECRET_KEY=config('OPENAI_API_KEY')

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     task="text-generation",
#     # max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=config('huggingfacehub_api_token'),
# )

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)

'''
template = "Tell me a joke about {topic}"
prompt_template =PromptTemplate.from_template(template)
'''

prompt_template=ChatPromptTemplate.from_messages([
    SystemMessage(content= "You are a helpful assistant"),
    HumanMessage(content= "Tell me a joke about {topic}")
])
result=prompt_template.invoke({'topic':'cat'})
# print(result.to_messages())
# res=result.messages[1].content
res=llm.invoke(result.to_messages()[1].content)
print(res)
"""
 res=result.messages[1].content
    result.to_messages()[1].content

    are equal
"""