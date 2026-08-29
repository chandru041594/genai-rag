import streamlit as st
from ingestion import ingest_pdf, ingest_docx, ingest_csv
from embeddings import add_to_index, retrieve, index
from transformers import pipeline

st.title("Gen AI RAG Assistant (Cloud Ready)")

# --- Load Hugging Face model ---
# You can swap this for another free model if needed
qa_model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# --- Document Upload ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "csv"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = ingest_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = ingest_docx(uploaded_file)
    elif uploaded_file.type == "text/csv":
        text = ingest_csv(uploaded_file)
    else:
        text = None

    if text:
        add_to_index(text, {"filename": uploaded_file.name})
        st.success(f"✅ Added {uploaded_file.name} to FAISS index")

# --- Question Input ---
query = st.text_input("Ask a question:")
if query:
    if index is None:
        st.error("⚠️ Please upload a document first before asking questions.")
    else:
        # Retrieve relevant context
        context = retrieve(query)

        # Build prompt
        prompt = f"Answer only from context:\n{context}\n\nQuestion: {query}\nIf unknown, say 'I don't know'."

        # ✅ Hugging Face pipeline call
        response = qa_model(prompt, max_length=512, do_sample=False)[0]["generated_text"]

        # Show response
        st.write(response)
