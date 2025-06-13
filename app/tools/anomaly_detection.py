from langchain.chains import RetrievalQA
from app.embeddings.embedding_manager import get_vectorstore
from langchain_community.chat_models import ChatOllama
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os

from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def draw_wrapped_text(canvas_obj, text, x=50, y=800, width=500, height=700, font_size=11):
    """
    Draws wrapped, paginated text using ReportLab's Platypus engine.
    """
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'Custom',
        parent=styles['Normal'],
        fontName="DejaVu",
        fontSize=font_size,
        leading=font_size + 3,
        alignment=TA_LEFT,
    )

    paragraphs = text.strip().split('\n')
    story = [Paragraph(p.strip(), custom_style) for p in paragraphs if p.strip()]
    
    frame = Frame(x, y - height, width, height, showBoundary=0)
    frame.addFromList(story, canvas_obj)

def anomaly_detection(company_name: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = ChatOllama(model="mistral")

    prompt = f"""
You are a forensic financial auditor. Analyze the financial data of **{company_name}** for statistical anomalies or inconsistencies.

Respond with insights like:
- 🧾 Unusual 400% marketing spend in Q2 with no associated revenue growth.
- ⚠️ Spikes or drops in ratios that aren't justified.
- ✅ No anomalies detected in core financial columns.
"""

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(prompt)

    output_dir = "output/anomaly_reports"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Anomaly_Report_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Register font
    font_path = "fonts/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

    # Create canvas
    c = canvas.Canvas(filepath, pagesize=A4)
    c.setFont("DejaVu", 11)

    # Draw wrapped & paginated text
    width, height = A4
    draw_wrapped_text(c, result, x=50, y=height - 50, width=width - 100, height=height - 100)

    c.save()
    return f"✅ Anomaly detection report saved to: {filepath}"
