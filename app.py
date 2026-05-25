"""
DocuMind — LLM-powered Document Q&A
Uses Claude API with a manual RAG pipeline (no LangChain).
"""

import os
import re
import math
import streamlit as st
from anthropic import Anthropic

# ─────────────────────────────────────────
# RAG UTILITIES  (zero external vector DB)
# ─────────────────────────────────────────

def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping windows."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def tfidf_score(query: str, chunk: str) -> float:
    """
    Lightweight TF-IDF-style similarity — no external libs.
    We treat each query token as a 'term' and compute term frequency
    in the chunk, weighted by inverse document frequency approximated
    by token length (rare long words score higher).
    """
    q_tokens = set(re.findall(r"\w+", query.lower()))
    c_tokens  = re.findall(r"\w+", chunk.lower())
    if not c_tokens:
        return 0.0
    score = 0.0
    for tok in q_tokens:
        tf = c_tokens.count(tok) / len(c_tokens)
        idf = math.log(1 + len(tok))          # longer tokens = rarer = higher IDF
        score += tf * idf
    return score


def retrieve(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Return top_k most relevant chunks for a query."""
    scored = [(tfidf_score(query, c), c) for c in chunks]
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .stChatMessage { border-radius: 12px; }
  h1 { font-family: Georgia, serif; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────
with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("Ask questions about your own documents using Claude.")

    api_key = st.text_input("Anthropic API Key", type="password",
                             value=os.getenv("ANTHROPIC_API_KEY", ""))

    st.divider()
    uploaded = st.file_uploader("Upload a document", type=["txt", "pdf"])

    if uploaded:
        if uploaded.type == "application/pdf":
            try:
                import pdfplumber
                with pdfplumber.open(uploaded) as pdf:
                    raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                st.error("Install pdfplumber: `pip install pdfplumber`")
                raw_text = ""
        else:
            raw_text = uploaded.read().decode("utf-8", errors="ignore")

        st.session_state["chunks"] = chunk_text(raw_text)
        st.success(f"✅ {len(st.session_state['chunks'])} chunks indexed")
        st.caption(f"~{len(raw_text):,} characters loaded")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Document → split into 500-word overlapping chunks  
2. Your question → TF-IDF scores every chunk  
3. Top 3 chunks injected as context into Claude's prompt  
4. Full conversation history keeps multi-turn coherent  
    """)

# ── Session state ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ── Chat area ─────────────────────────────
st.header("Chat with your document")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the document…"):
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()
    if not st.session_state.chunks:
        st.error("Please upload a document first.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── RAG: retrieve relevant context ────
    relevant_chunks = retrieve(prompt, st.session_state.chunks, top_k=3)
    context_block = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document excerpts.
If the answer is not in the excerpts, say so honestly — do not invent information.

DOCUMENT EXCERPTS:
{context_block}
"""

    # ── Call Claude ────────────────────────
    client = Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Build messages (full history for multi-turn)
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=api_messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show which chunks were used (expandable)
    with st.expander("🔍 Chunks used as context"):
        for i, chunk in enumerate(relevant_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk[:300]}…")
