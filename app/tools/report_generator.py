from langchain.chains import RetrievalQA
from app.embeddings.embedding_manager import get_vectorstore
from langchain_community.chat_models import ChatOllama
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os

def generate_financial_report(company_name: str) -> str:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(model="mistral")

    prompt = f"""
You are a financial analyst assistant. Generate a structured financial analysis report for the company: {company_name}.

Use the following format:
---
# 📊 Financial Report: {company_name}

## 🧾 Executive Summary

## 💡 Key Financial Highlights

## ⚠️ Risk & Red Flags

## 📈 Market Sentiment (if available)

## 📌 Conclusion
---

Use information only from reliable financial documents or filings. Keep the tone professional and clear.
"""

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(prompt)

    # ---- Generate PDF using reportlab ----
    output_dir = "output/report"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{company_name.replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Make sure you have 'fonts/DejaVuSans.ttf'
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

    return f"✅ Financial report for '{company_name}' generated and saved to: {filepath}"
