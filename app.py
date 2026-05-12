import streamlit as st
import numpy as np
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi

from langchain_groq import ChatGroq

# =========================
# GROQ API KEY (PASTE HERE)
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Hybrid RAG Chatbot")

st.title("⚡ Hybrid + Vectorless RAG Chatbot")

# =========================
# SESSION MEMORY
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False

# =========================
# MODE SELECTION
# =========================
mode = st.radio(
    "Choose Mode",
    ["🌐 AI Chat Mode", "📄 Document Mode", "🔀 Hybrid Mode"]
)

# =========================
# PDF UPLOAD
# =========================
uploaded_file = None
if mode != "🌐 AI Chat Mode":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# =========================
# PROCESS PDF
# =========================
if uploaded_file and not st.session_state.docs_ready:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    # Vectorless RAG (BM25)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Semantic RAG
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    # Store in session
    st.session_state.chunks = chunks
    st.session_state.texts = texts
    st.session_state.bm25 = bm25
    st.session_state.model = model
    st.session_state.embeddings = embeddings
    st.session_state.docs_ready = True

    st.success("PDF Loaded Successfully!")

# =========================
# USER INPUT
# =========================
query = st.text_input("Ask your question")

# =========================
# MAIN LOGIC
# =========================
if query:

    response = ""

    # -------------------------
    # 🌐 AI MODE
    # -------------------------
    if mode == "🌐 AI Chat Mode":

        response = llm.invoke(query).content

    # -------------------------
    # 📄 DOCUMENT MODE
    # -------------------------
    elif mode == "📄 Document Mode":

        if not st.session_state.docs_ready:
            response = "Please upload a PDF first!"
        else:

            bm25 = st.session_state.bm25
            model = st.session_state.model
            embeddings = st.session_state.embeddings
            chunks = st.session_state.chunks

            # BM25
            scores = bm25.get_scores(query.lower().split())
            top_bm25 = np.argsort(scores)[::-1][:3]
            bm25_docs = [chunks[i] for i in top_bm25]

            # Semantic
            q_emb = model.encode([query])
            sims = cosine_similarity(q_emb, embeddings)[0]
            top_sem = np.argsort(sims)[::-1][:3]
            sem_docs = [chunks[i] for i in top_sem]

            combined = bm25_docs + sem_docs

            seen = set()
            final_docs = []

            for d in combined:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    final_docs.append(d)

            context = "\n\n".join([d.page_content for d in final_docs[:5]])

            prompt = f"""
Answer ONLY from the document:

{context}

Question: {query}
"""

            response = llm.invoke(prompt).content

    # -------------------------
    # 🔀 HYBRID MODE
    # -------------------------
    else:

        if st.session_state.docs_ready:

            bm25 = st.session_state.bm25
            model = st.session_state.model
            embeddings = st.session_state.embeddings
            chunks = st.session_state.chunks

            # BM25
            scores = bm25.get_scores(query.lower().split())
            top_bm25 = np.argsort(scores)[::-1][:3]
            bm25_docs = [chunks[i] for i in top_bm25]

            # Semantic
            q_emb = model.encode([query])
            sims = cosine_similarity(q_emb, embeddings)[0]
            top_sem = np.argsort(sims)[::-1][:3]
            sem_docs = [chunks[i] for i in top_sem]

            combined = bm25_docs + sem_docs

            seen = set()
            final_docs = []

            for d in combined:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    final_docs.append(d)

            context = "\n\n".join([d.page_content for d in final_docs[:5]])

            prompt = f"""
Use document if relevant, otherwise answer normally.

Document:
{context}

Question: {query}
"""

            response = llm.invoke(prompt).content

        else:
            response = llm.invoke(query).content

    # =========================
    # SAVE MEMORY
    # =========================
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Bot", response))

    # =========================
    # SHOW ANSWER FIRST
    # =========================
    st.subheader("Answer")
    st.write(response)

    # =========================
    # MEMORY AT BOTTOM
    # =========================
    st.markdown("---")
    st.subheader("🧠 Chat Memory")

    for role, msg in st.session_state.chat_history:
        st.write(f"**{role}:** {msg}")

