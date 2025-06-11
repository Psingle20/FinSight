from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from app.embeddings.embedding_manager import get_vectorstore

def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOllama(model="llama3")
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
