import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# -------------------------------------------------------
# 1. Load Speech Text
# -------------------------------------------------------
def load_text(file_path="speech.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"speech.txt not found in {os.getcwd()}")
    print("Loading speech text...")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------------
# 2. Split Text into Chunks
# -------------------------------------------------------
def split_into_chunks(text):
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    return splitter.create_documents([text])


# -------------------------------------------------------
# 3. Initialize Embeddings + Chroma Vector Store
# -------------------------------------------------------
def create_vectorstore(chunks, persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        collection_name="ambedkar_speech",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    doc_count = vector_store._client.get_collection("ambedkar_speech").count()

    if doc_count > 0:
        print(f"Vector store already contains {doc_count} documents. Skipping re-insert.")
    else:
        print("Creating Chroma vector store and inserting documents...")
        vector_store.add_documents(chunks)
        print("Documents inserted successfully.")

    return vector_store


# -------------------------------------------------------
# 4. Retrieve Relevant Chunks
# -------------------------------------------------------
def retrieve_chunks(query, vector_store, k=3):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)


# -------------------------------------------------------
# 5. Use Ollama Mistral to Generate Answer
# -------------------------------------------------------
def answer_question(question, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
You are a Q&A assistant. You MUST answer ONLY using the provided context.
If something is not present in the context, say "The context does not contain this information."

Context:
{context}

Question: {question}

Answer:
"""
    llm = ChatOllama(model="mistral")
    response = llm.invoke(prompt)
    return response.content.strip()


# -------------------------------------------------------
# Interactive CLI
# -------------------------------------------------------
def run_cli():
    
    text = load_text()
    chunks = split_into_chunks(text)
    vector_store = create_vectorstore(chunks)

    print("System ready. Ask questions about the speech.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting cli...")
            break

        retrieved_docs = retrieve_chunks(query, vector_store)
        answer = answer_question(query, retrieved_docs)
        
        print("\n--------------------")
        print("\nAssistant:", answer)
        print("\n--------------------")


if __name__ == "__main__":
    run_cli()
