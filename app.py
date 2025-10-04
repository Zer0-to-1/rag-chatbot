# app.py — Baby Steps RAG Chatbot
import os
import uuid
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from pypdf import PdfReader

# --------------- Simple config ---------------
PERSIST_DIR = "vectordb"
MODEL_NAME = "mistral"   # Change in sidebar if you like
EMB_NAME = "all-MiniLM-L6-v2"  # small, fast embedding model

# --------------- Init ---------------
st.set_page_config(page_title="RAG Chatbot (Baby Steps)", page_icon="🍼", layout="wide")
st.title("🍼 RAG Chatbot — Baby Steps Edition")

# Create embeddings + Chroma DB
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMB_NAME)

@st.cache_resource
def get_chroma():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = Client(Settings(persist_directory=PERSIST_DIR))
    return client.get_or_create_collection("docs_baby_steps")

EMB = get_embedder()
COL = get_chroma()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, assistant)

# --------------- Helpers ---------------
def chunk_text(text, chunk_size=900, overlap=150):
    text = text.replace("\n", " ").strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start <= 0:
            break
    return chunks

def embed_texts(texts):
    return EMB.encode(texts, convert_to_numpy=True).tolist()

def add_documents(docs, source_name):
    # docs = list of strings
    embeddings = embed_texts(docs)
    ids = [str(uuid.uuid4()) for _ in docs]
    metas = [{"source": source_name, "chunk": i} for i in range(len(docs))]
    COL.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

def retrieve(query, k=4):
    q_emb = embed_texts([query])[0]
    res = COL.query(query_embeddings=q_emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def call_ollama(prompt, model):
    # Uses Ollama local server: POST /api/generate
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"(Error talking to Ollama: {e})\n\nDid you run `ollama pull {model}` and is Ollama running?"

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model name", value=MODEL_NAME, help="e.g., mistral, llama3, llama2, qwen, etc.")
    top_k = st.slider("Top‑K retrieved chunks", 2, 8, 4)
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.success("Cleared!")

# --------------- Upload PDFs ---------------
st.subheader("1) Upload your PDFs")
uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        reader = PdfReader(f)
        all_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                all_text.append(t)
        if not all_text:
            st.warning(f"Couldn't extract text from {f.name}.")
            continue
        text = "\n".join(all_text)
        chunks = chunk_text(text)
        add_documents(chunks, source_name=f.name)
        st.success(f"Indexed {len(chunks)} chunks from {f.name}.")

# --------------- Ask a question ---------------
st.subheader("2) Ask a question about your PDFs")
question = st.text_input("Type your question and press Enter")

if question:
    # Retrieve
    ctx = retrieve(question, k=top_k)
    context = "\n\n".join([f"[Source: {m.get('source')} | Chunk: {m.get('chunk')}] {c}" for c, m in ctx])

    # Build prompt
    system_note = (
        "You are a helpful assistant. Answer ONLY from the context. "
        "If the answer is not in the context, say you don't know."
    )
    prompt = f"{system_note}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"

    # Generate
    answer = call_ollama(prompt, model=model)

    # Save & show
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))

    st.markdown("### 💬 Answer")
    st.write(answer)

    st.markdown("### 🔎 Sources (chunks used)")
    for i, (c, m) in enumerate(ctx, start=1):
        st.markdown(f"**{i}. {m.get('source')} (chunk {m.get('chunk')})**")
        st.caption(c[:500] + ("…" if len(c) > 500 else ""))

# --------------- Conversation history ---------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🗂️ Conversation")
    for role, msg in st.session_state.history[-8:]:  # show last 8 turns
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
