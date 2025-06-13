from langchain.chains import RetrievalQA
from app.embeddings.embedding_manager import get_vectorstore
from langchain_community.chat_models import ChatOllama
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os

def anomaly_detection(company_name: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = ChatOllama(model="mistral")

    prompt = """
You are a forensic financial auditor. Analyze the data for any statistical anomalies or inconsistencies in financial reports or statements.

Respond with insights like:
- 🧾 Unusual 400% marketing spend in Q2 with no associated revenue growth.
- ⚠️ Spikes or drops in ratios that aren't justified.
- ✅ No anomalies detected in core financial columns.
"""

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(prompt)

    output_dir = "output/anomaly_reports"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Anomaly_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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
    return f"✅ Anomaly detection report saved to: {filepath}"
