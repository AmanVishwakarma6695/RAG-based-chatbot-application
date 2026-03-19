from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

persistent_directory = "db\chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"


retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print("--------------------------------------")
print(f"User Query: {query}")
print("--------------------------------------")
        
# Display results
print("Answer retrieved from the vector store:")
       
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

print("--------------------------------------")

#{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

docs_text = ""
for doc in relevant_docs:
    docs_text += "- " + doc.page_content + "\n"

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{docs_text}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

print("--------------------------------------")

print(f"The Combined Input is : {combined_input}")

print("--------------------------------------")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

response = llm.invoke(messages)


# Display the full result and content only
print("--------------------------------------")
print("\n--- Generated Response ---")
print("Content only:")
print(response.content)





#Questions to test the retriever
# 1. How many Skilled QA Professionals does Jignect has?
# 2. What are the services provided by Jignect?
# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"