from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
documents = []

def add_to_index(text, metadata):
    global index, documents
    embedding = model.encode([text])[0]
    if index is None:
        index = faiss.IndexFlatL2(len(embedding))
    index.add(np.array([embedding]))
    documents.append({"text": text, "metadata": metadata})

def retrieve(query, k=3):
    q_emb = model.encode([query])[0]
    D, I = index.search(np.array([q_emb]).astype("float32"), k)
    return [documents[i]["text"] for i in I[0]]
