from decouple import config
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import psycopg2
import uuid
from datetime import datetime

# PostgreSQL connection details
DB_NAME = config("DB_NAME")
DB_USER = config("DB_USER")
DB_PASSWORD = config("DB_PASSWORD")
DB_HOST = config("DB_HOST", default="localhost")
DB_PORT = config("DB_PORT", default="5432")

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
cur = conn.cursor()

session_id=str(uuid.uuid4()).split("-")[0]
# Function to insert chat messages
def save_message(role, content):
    cur.execute("INSERT INTO chat_messages (session_id,role,content) VALUES (%s,%s, %s)", (session_id,role,content ))
    conn.commit()

# Function to retrieve chat history
def get_chat_history():
    cur.execute("SELECT role, content FROM chat_messages ORDER BY timestamp")
    return cur.fetchall()


# Function to create table if not exists
def create_table():
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(15) NOT NULL,
            role VARCHAR(10) CHECK (role IN ('system', 'user', 'ai')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

# Create the table if it doesn't exist
create_table()

SECRET_KEY=config('OPENAI_API_KEY')
# Create a ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=SECRET_KEY
    
)
chat_history = []  # Use a list to store messages


# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

save_message("system", system_message.content)
# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message
    save_message("user", query)  # Store user message in DB
    # Get AI response using history
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    save_message("ai", response)  # Store AI response in DB
    print(f"AI: {response}")




# Close database connection
cur.close()
conn.close()