import os

from decouple import config
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SECRET_KEY=config('OPENAI_API_KEY')

# Define the persistent directory
current_dir=os.getcwd()
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=SECRET_KEY)

# Load the existing vector store with the embedding function
db=Chroma(
        collection_name="lord_of_the_rings_collection",
        embedding_function=embeddings,
        persist_directory=persistent_directory
)

# Define the user's question
query = "What does dracula fear the most?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# # Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=SECRET_KEY)

# # Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# # Invoke the model with the combined input
result = model.invoke(messages)

# # Display the full result and content only
# print("\n--- Generated Response ---")
# # print("Full result:")
# print(result)
# print("Content only:")
print(result.content)