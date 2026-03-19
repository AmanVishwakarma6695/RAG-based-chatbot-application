from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_documents(docs_path="docs"):
    print(f"Loading Documnets from {docs_path}")

    #to check path exist or not
    if not os.path.exists(docs_path):
        print("--------------------------------------")
        raise FileNotFoundError(f"The folder {docs_path} doesnot exist. Please create the folder.")
        print("--------------------------------------")


    # Main code to load the documents from the specified path

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    # Just to check that documents are loaded properly and not empty
    if (len(documents) == 0):
        raise ValueError(f"No .txt files found in {docs_path}. Please add some .txt files to the folder.")
    
    # Show some basic information about the loaded documents
    # for i, doc in enumerate(documents[:2]):  # Show first 2 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")
    
    return documents


def split_documnet(documnets, chunk_size=800, chunk_overlap=0):

    # Main code to split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documnets)


    #Just to check that Chunk is not Empty
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory='db/chroma_db'):
    print("----------------------------------------------")
    print("Creating embeddings and storing in ChromaDB...")
    print("----------------------------------------------")


    # print("----------------------------------------------")
    # API_KEY = os.getenv("GOOGLE_API_KEY")

    # print(f"Using Google API Key: {API_KEY}")

    # embedding_model = GoogleGenerativeAIEmbeddings(
    #                     model="models/gemini-embedding-001",
    #                     google_api_key=os.getenv("GOOGLE_API_KEY"))
    

    # Main code to create the vector store using Chroma

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Embedding Model Cleared")
    print("----------------------------------------------")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Just to check that vector store is created properly and not empty
    print("--- Finished creating vector store ---")    
    print(f"Vector store created and saved to {persist_directory}")
    print("----------------------------------------------")

    return vector_store


def main():

    docs_location = "docs"
    print("Main Function starts here")

    # Calling def to load the documents from the specified path
    document = load_documents(docs_location)

    # Calling def to split the document into chunks
    chunks = split_documnet(document)

    # Calling def to create the vector store using Chroma
    vector_store = create_vector_store(chunks)


if __name__ == "__main__":
    main()
