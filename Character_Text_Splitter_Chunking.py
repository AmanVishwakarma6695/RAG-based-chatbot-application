import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Example text to split into chunks
tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# Create a CharacterTextSplitter
def character_text_splitter():
    print("This is Character Text Splitter and Chunking")
    text_splitter = CharacterTextSplitter(
                    chunk_size=100, 
                    chunk_overlap=0,
                    separator="\n\n")
    
    # Split the text into chunks
    chunks = text_splitter.split_text(tesla_text)

    # Print the chunks
    for i, chunks in enumerate(chunks, 1):
        print("--------------------------------------")
        print(f"\n--- Chunk {i} ---")
        print(len(chunks), "characters")
        print(chunks)
        print("--------------------------------------")

# Create a CharacterTextSplitter
def recursive_character_text_splitter():
    print("This is Recursive Character Text Splitter and Chunking")
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=100, 
                    chunk_overlap=0,
                    separators=["\n\n", "\n", ".", " ", ""])
    
    # Split the text into chunks
    chunks = text_splitter.split_text(tesla_text)

    # Print the chunks
    for i, chunks in enumerate(chunks, 1):
        print("--------------------------------------")
        print(f"\n--- Chunk {i} ---")
        print(len(chunks), "characters")
        print(chunks)
        print("--------------------------------------")


def main():
    user_input = input("Enter 1 to run Character Text Splitter or 2 to run Recursive Character Text Splitter: ")

    if user_input == "1":
        character_text_splitter()
    
    else:
        recursive_character_text_splitter()

#This means : Run the chat only if this file is executed directly. If this file is imported as a module in another file, the code inside this block will not be executed.
if __name__ == "__main__":
    main()