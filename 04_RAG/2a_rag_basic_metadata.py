from decouple import config
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import os 
from langchain_chroma import Chroma
import chromadb
SECRET_KEY=config('OPENAI_API_KEY')

# Define the directory containing the text file and the persistent directory
# current_dir=os.path.dirname(os.path.abspath('__file__'))
current_dir=os.getcwd()
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir,"chroma_db_with_metadata")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure the text file exists.
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check the file path."
        )
    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
   
    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path,encoding="utf-8")
        book_docs = loader.load()
        
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)
            
    
   
    # # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # # Display information about the split documents
    print("\n---Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # # Create embeddings
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

    # # Optionally, add the documents to the collection if required by your workflow
    db.add_documents(docs)
    # print(db)
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
    query="Where is Dracula's castle located?"

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k":4,'score_threshold': 0.2}
    )
    retriever_data=retriever.invoke(query)
    # print(retriever_data)
    for index,doc in enumerate(retriever_data):
        print(f"Document {index}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
