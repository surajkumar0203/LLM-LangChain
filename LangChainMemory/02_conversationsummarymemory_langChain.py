from langchain_openai import ChatOpenAI
from decouple import config
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

SECRET_KEY=config('OPENAI_API_KEY')



llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY,
    temperature=0,

)
memory = ConversationSummaryMemory(llm=llm)

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
# print()
'''
    ConversationSummaryMemory me jo conversation hota hai use sumrize kar diya jata internally LLM ke through.
    
    Current conversation:
    Suraj Kumar introduces himself to the AI, who responds by greeting Suraj and offering assistance. Suraj asks how the AI is doing, and the AI responds that it's been busy processing information and learning new things. The AI then asks how it can assist Suraj.
    Human: who am I?
    AI:
'''

# see conversation history
print(memory.load_memory_variables({}))
print(conversation.memory.buffer)