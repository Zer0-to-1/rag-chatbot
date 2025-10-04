# 🍼 RAG Chatbot — Baby Steps Edition (Windows/Mac/Linux)

This is a **beginner‑friendly** project that lets you build a working **AI chatbot** that can answer questions about **your own PDFs**.
You do **not** need deep ML knowledge. Just follow the steps exactly.

---

## 🧠 What is RAG?
**R**etrieval‑**A**ugmented **G**eneration = The chatbot first **retrieves** relevant text from your documents, then the AI model **generates** an answer using that context.
Think of it as an **open‑book exam**: the bot checks your PDFs before answering.

---

## ✅ What you’ll get
- A local chatbot you can run on your laptop.
- Upload PDFs and ask questions. The bot answers **with sources**.
- Clean, minimal **Streamlit UI**.
- No paid APIs. Uses a **free local model** via **Ollama**.

---

## 🛠️ What you need (install once)
1) **Python 3.10+** (Windows/Mac/Linux).  
2) **Git** (optional, nice to have).  
3) **Ollama** (runs small open models locally): install from the official Ollama website, then **open it** so it’s running in the background.

> After installing Ollama, open **Terminal / PowerShell** and run **one** of these to download a model:
```
ollama pull mistral     # good default
# or
ollama pull llama3      # if available on your machine
```

---

## ▶️ How to run (copy–paste)
Open **PowerShell (Windows)** or **Terminal (Mac/Linux)** and run:

```bash
# 1) Go to the project folder
cd rag_chatbot_baby_steps

# 2) (Optional but recommended) Create a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

# 3) Install Python packages
pip install -r requirements.txt

# 4) Start the app
streamlit run app.py
```

Then open the browser tab that appears (usually http://localhost:8501).

---

## 🧾 How to use the app
1) Make sure **Ollama** is running (you installed it and it’s open).
2) In the app, **upload one or more PDFs** (syllabus, notes, manuals, etc.).
3) Ask a question in the text box (e.g., “What are the deadlines?”).
4) The bot shows an answer **and the exact source chunks** it used.

---

## 📁 Project structure
```
rag_chatbot_baby_steps/
├─ app.py                # Streamlit UI + RAG pipeline
├─ requirements.txt      # Python dependencies
├─ README.md             # This file (step-by-step guide)
└─ vectordb/             # (auto-created) where your document embeddings live
```

---

## 🧩 How it works (simple)
- We split your PDFs into small **chunks** of text.
- We convert each chunk into a number vector (“**embedding**”) using **Sentence Transformers**.
- We store vectors in a small database (**Chroma**).
- When you ask a question: we embed your question → **find the most similar chunks** →
  build a **prompt** with those chunks → ask the local model in **Ollama** to answer.
- We show the answer **and** the sources.

---

## 🧯 Common fixes
- **Model not found** → run `ollama pull mistral` in Terminal/PowerShell.
- **Ollama not running** → open the Ollama app OR run `ollama serve`.
- **GPU errors** → Ollama will fall back to CPU; it’s slower but fine for learning.
- **Long PDFs** → you can upload multiple, but start small to keep it snappy.

---

## 🙏 Credits
- Local LLM runtime powered by **Ollama**.
- Text embeddings by **sentence-transformers** (`all-MiniLM-L6-v2`).
- Vector DB is **Chroma**.
- UI built with **Streamlit**.

