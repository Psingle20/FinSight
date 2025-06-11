# from app.agents.rag_agent import get_rag_chain

# def run_cli():
#     qa_chain = get_rag_chain()
#     print("🔎 Financial Analyst Assistant Ready. Type 'exit' to quit.")
    
#     while True:
#         query = input("💬 You: ")
#         if query.lower() in ["exit", "quit"]:
#             break
#         response = qa_chain.run(query)
#         print(f"🤖 Bot: {response}\n")



from app.agents.rag_agent import get_agent  # Change this line

def run_cli():
    agent = get_agent()  # This now returns the tool-enabled agent
    print("🔎 Financial Analyst Assistant Ready. Type 'exit' to quit.")
    
    while True:
        query = input("💬 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)  # Agent handles tool invocation or RAG
        print(f"🤖 Bot: {response}\n")
