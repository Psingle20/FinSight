# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOllama
# from app.embeddings.embedding_manager import get_vectorstore
# from langchain.agents import Tool
# from app.tools.report_generator import generate_financial_report


# def get_rag_chain():
#     vectorstore = get_vectorstore()
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
#     llm = ChatOllama(model="llama3")
    
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=False
#     )
from app.tools.risk_assessment import risk_assessment
from app.tools.anomaly_detection import anomaly_detection
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from app.embeddings.embedding_manager import get_vectorstore
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from app.tools.report_generator import generate_financial_report

def get_rag_tool():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return Tool(
        name="retrieval_qa",
        func=qa_chain.run,
        description="Use this tool by name 'retrieval_qa' to answer questions based on financial documents."
    )

def get_report_tool():
    return Tool(
        name="generate_financial_report",
        func=generate_financial_report,
        description="Generates a detailed financial report based on document provided and user query provide the company name the document belongs to as input."
    )

def get_risk_assessment_tool():
    return Tool(
        name="generate_risk_assessment",
        func=risk_assessment,
        description="Analyzes financial PDFs for risk (liquidity, credit, market). Input: company name."
    )

def get_anomaly_detection_tool():
    return Tool(
        name="detect_anomalies",
        func=anomaly_detection,
        description="Detects financial anomalies in PDF data. Input: company name."
    )


def get_agent():
    tools = [
        get_rag_tool(),
        get_report_tool(),
        get_risk_assessment_tool(),
        get_anomaly_detection_tool()
    ]
    llm = ChatOllama(model="mistral")

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
