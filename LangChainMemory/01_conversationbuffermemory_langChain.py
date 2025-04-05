from langchain_openai import ChatOpenAI
from decouple import config
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

SECRET_KEY=config('OPENAI_API_KEY')

memory = ConversationBufferMemory()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY,
    temperature=0,

)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)


res=conversation.predict(input="Hi there! I am Suraj Kumar!")
print(res)
res=conversation.predict(input="How are you today?")
print(res)
res=conversation.predict(input="who am I?")
print(res)
print()
'''
    ConversationBufferMemory me jo conversation hua hai wo same as it store hota hai. Sirf append hota hai.
    no change in memory
    Example:
    
    Current conversation:
    Human: Hi there! I am Suraj Kumar!
    AI: Hello Suraj Kumar! It's nice to meet you. How can I assist you today?
    Human: How are you today?
    AI: I'm doing well, thank you for asking! I'm here and ready to help with any questions or tasks you have. What can I assist you with today, Suraj Kumar?
    Human: who am I?
    AI:

    Disadvantage :
    Imagine 
    jo bhi convertion hota hai wo memory me aata hai phir execute hota hai jo result hota hai . jo result hota hai wo show
    hota hai.
    Par ek samaye aisa bhi hoga context ka size bahut bara ho jayega. kisi bhi LLM ka limit token hota hai. ya token ka paisa bhi
    lagta hai.
'''

# see conversation history
# print(memory.load_memory_variables({}))
print(conversation.memory.buffer)