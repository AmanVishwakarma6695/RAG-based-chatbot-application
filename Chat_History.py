from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

persistent_directory = "db\chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"} )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Store our conversation as messages
chat_history = []

def ask_question(your_query):
    print(f"\n--- You asked: {your_query} ---")
    
    #Step 1 : Rewrite the question to be standalone and searchable using the chat history
    if chat_history:
        messages = [SystemMessage(content = "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
                    ] + chat_history + [HumanMessage(content= f"New QUestion : {your_query}")]

        llm_question = llm.invoke(messages)
        final_question = llm_question.content
        print("--------------------------------------")
        print(f"Rewritten Question: {final_question}")
        print("--------------------------------------")
    else :
        final_question = your_query



    #Step 2 : Retrieve relevant documents from the vector store using the rewritten question
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(final_question)
    # Display results
    print("Answer retrieved from the vector store:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
    print("--------------------------------------")


    #Step 3 : Create final input for the LLM by combining the retrieved documents and the original question, then generate a response
    docs_text = ""
    for doc in relevant_docs:
        docs_text += "- " + doc.page_content + "\n"

    # Combine the query and the relevant document contents
    combined_input = f"""Based on the following documents, please answer this question: {final_question}

    Documents:
    {docs_text}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    #Step 4 : Get the answer
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

    #step 5 : Add the question and answer to the chat history
    chat_history.append(HumanMessage(content=final_question))
    chat_history.append(AIMessage(content=response.content))

    return response


def start_chat():
    print("Welcome to the Chat History Retrieval System!")
    print("You can ask questions and type 'exit' to quit.")

    while True:
        print("--------------------------------------")
        user_query = input("Enter your question: ")
        print("--------------------------------------")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        ask_question(user_query)


if __name__ == "__main__":
    start_chat()