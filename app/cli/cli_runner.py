from app.agents.rag_agent import get_rag_chain

def run_cli():
    qa_chain = get_rag_chain()
    print("🔎 Financial Analyst Assistant Ready. Type 'exit' to quit.")
    
    while True:
        query = input("💬 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.run(query)
        print(f"🤖 Bot: {response}\n")
