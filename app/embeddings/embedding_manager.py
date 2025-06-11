from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.loaders.pdf_loader import load_documents
import os

def get_vectorstore():
    persist_dir = "vector_store"

    # Use HuggingFaceEmbeddings with a lightweight model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing vectorstore if available
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("[DEBUG] Loading existing vector store from disk...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    print("[DEBUG] No existing vector store found. Loading documents...")
    documents = load_documents()
    if not documents:
        raise ValueError("[ERROR] No documents loaded. Please check the data folder.")

    print(f"[DEBUG] Loaded {len(documents)} documents. Creating new vector store...")
    vs = Chroma.from_documents(documents, embedding, persist_directory=persist_dir)
    vs.persist()
    print("[DEBUG] Vector store created and persisted.")
    print(f"[DEBUG] Vectorstore has {len(vs.get())} documents.")

    return vs
