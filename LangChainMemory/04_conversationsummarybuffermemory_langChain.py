from langchain_openai import ChatOpenAI
from decouple import config
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

SECRET_KEY=config('OPENAI_API_KEY')



llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY,
    temperature=0,

)
memory = ConversationSummaryBufferMemory(llm=llm,max_token_limit=3)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)


res=conversation.predict(input="Hi there! I am Suraj Kumar!")
print(res)
res=conversation.predict(input="How are you today?")
print(res)
res=conversation.predict(input="Can you write small artical to post on LinkedIn")
print(res)
res=conversation.predict(input="who am I?")
print(res)
# print()
'''
    ConversationSummaryBufferMemory this is hybrid model  work token bases

    Current conversation:
    System: Suraj Kumar introduces themselves to the AI, who responds warmly and offers assistance. Suraj Kumar asks how the AI is doing, to which the AI replies that it is well and ready to help with any questions or tasks. Suraj Kumar then asks the AI to write a small article to post on LinkedIn, to which the AI enthusiastically agrees. The AI offers to generate a draft based on specific topic and key points provided by Suraj Kumar, allowing for review and editing before posting.
    Human: who am I?
    AI:
'''

# see conversation history
print(memory.load_memory_variables({}))
print(conversation.memory.buffer)