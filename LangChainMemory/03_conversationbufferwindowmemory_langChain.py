from langchain_openai import ChatOpenAI
from decouple import config
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

SECRET_KEY=config('OPENAI_API_KEY')



llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY,
    temperature=0,

)
memory = ConversationBufferWindowMemory(k=1)

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
    ConversationBufferWindowMemory isme k=2 hai to 2 conversation tak store rakhega.
    


    When k=1
    Current conversation:
Human: Can you write small artical to post on LinkedIn
AI: Of course, I can help with that! What specific topic would you like the article to be about? Do you have any key points or ideas you would like me to include in the article? Let me know so I can tailor the content to your preferences.
Human: who am I?
AI:

when i ask who am I?
I'm sorry, I do not have access to personal information about individuals. If you would like to share more details about yourself, I can help you create a professional bio or summary for your LinkedIn profile. Just let me know what you would like to highlight about yourself!
'''

# see conversation history
print(memory.load_memory_variables({}))
print(conversation.memory.buffer)