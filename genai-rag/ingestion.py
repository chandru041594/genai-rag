import pdfplumber, docx, pandas as pd

def ingest_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def ingest_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def ingest_csv(file):
    df = pd.read_csv(file)
    return df.to_string()
