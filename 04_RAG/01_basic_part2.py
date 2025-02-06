from decouple import config
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import os 
from langchain_chroma import Chroma

SECRET_KEY=config('OPENAI_API_KEY')

# Define the directory containing the text file and the persistent directory
# current_dir=os.path.dirname(os.path.abspath('__file__'))
current_dir=os.getcwd()

file_path = os.path.join(current_dir,"documents","lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir,"db","chroma_db")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure the text file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the file path."
        )
    
    # Read the text content from the file.
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    # print(documents)
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n---Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n---Creating Embeddings---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=SECRET_KEY
    )# Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")
    # print(embeddings)
    # Create the vector store and persist it automatically
    db=Chroma(
        collection_name="lord_of_the_rings_collection",
        embedding_function=embeddings,
        persist_directory=persistent_directory
    )
    # Optionally, add the documents to the collection if required by your workflow
    db.add_documents(docs)
    print(db)
else:
    print("Vector store already exist. No need to initialize")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=SECRET_KEY
    )

    db=Chroma(
        collection_name="lord_of_the_rings_collection",
        embedding_function=embeddings,
        persist_directory=persistent_directory
    )

    # Query
    query="Where does Gandalf meet Frodo?"

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k":10,'score_threshold': 0.9}
    )
    retriever_data=retriever.invoke(query)
    # print(retriever_data)

    for index,doc in enumerate(retriever_data):
        print(f"Document {index}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
