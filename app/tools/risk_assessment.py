from langchain.chains import RetrievalQA
from app.embeddings.embedding_manager import get_vectorstore
from langchain_community.chat_models import ChatOllama
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os

def risk_assessment(company_name: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = ChatOllama(model="mistral")

    prompt = f"""
You are a financial risk analyst. Analyze financial data for the company **{company_name}**.

Summarize risks in the following categories:
- Liquidity risk (e.g., current ratio, debt-to-equity)
- Market risk
- Credit risk (e.g., interest coverage)
- Operational red flags

Use this format:
⚠️ High leverage detected; D/E = 3.2  
✅ No major operational risk found.
"""

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(prompt)

    output_dir = "output/risk_reports"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Risk_Assessment_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)

    font_path = "fonts/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    c.setFont("DejaVu", 11)
    y = height - 50

    for line in result.split('\n'):
        if y < 50:
            c.showPage()
            c.setFont("DejaVu", 11)
            y = height - 50
        c.drawString(50, y, line)
        y -= 18

    c.save()
    return f"✅ Risk assessment report saved to: {filepath}"
