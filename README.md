# 📄 DocuMind — LLM Document Q&A

> **Ask questions about any document using Claude.** No vector databases, no LangChain — just a clean, from-scratch RAG pipeline you can understand line by line.

---

## What Problem It Solves

Knowledge workers waste hours searching through long PDFs, research papers, and reports for a single fact. DocuMind lets you **upload any document and have a multi-turn conversation with it** — getting precise, cited answers in seconds.

This is not a chatbot wrapper. It implements a complete Retrieval-Augmented Generation (RAG) pipeline from scratch, solving the core problem: *how do you give an LLM accurate knowledge of a document it has never seen, without hallucination?*

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                      │
│                                                            │
│   Upload (.txt/.pdf)  →  Chunker (500w, 80w overlap)       │
│                              ↓                             │
│                     In-memory chunk store                  │
└────────────────────────────────────────────────────────────┘
                              ↓  (at query time)
┌────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                         │
│                                                            │
│   User question  →  TF-IDF scorer  →  Top-3 chunks        │
│                              ↓                             │
│               Prompt assembler (context + history)         │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                   GENERATION (Claude API)                  │
│                                                            │
│   System: document excerpts injected as grounding context  │
│   Messages: full conversation history (multi-turn)         │
│   Model: claude-sonnet-4-20250514  (streaming)             │
│                              ↓                             │
│               Streamed answer to Streamlit UI              │
└────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | What it does |
|---|---|---|
| Chunker | `app.py → chunk_text()` | Splits document into 500-word windows with 80-word overlap to preserve sentence context across boundaries |
| Retriever | `app.py → retrieve()` | Scores every chunk against the query using a from-scratch TF-IDF function — no FAISS, no embeddings API needed |
| Prompt Builder | `app.py` (inline) | Injects the top-3 chunks as a `DOCUMENT EXCERPTS` block into Claude's system prompt |
| Multi-turn Memory | `st.session_state.messages` | Appends every user/assistant turn to the API call, giving the model full conversation context |
| Streaming | `client.messages.stream()` | Tokens are written to the UI as they arrive — feels instant |

---

## Engineering Challenge: Context Window Budgeting

The hardest unexpected problem was **fitting relevant content into the context window without degrading answer quality**.

**The naive approach** — sending the entire document as context — fails for two reasons:
1. Long documents exceed token limits
2. Irrelevant content confuses the model and increases hallucination rates

**First attempt:** Fixed 3-chunk retrieval. Problem: chunk boundaries sometimes cut sentences mid-thought, making the retrieved context incoherent.

**Solution implemented:**
- **Overlapping chunks** (80-word overlap) ensure no sentence is split across a boundary
- **TF-IDF retrieval** (not just keyword matching) weights rare, informative terms higher than common words — querying "What is the Turing Test?" correctly retrieves the 1950 context, not every paragraph that mentions "AI"
- A transparent "Chunks used" expander in the UI lets users audit exactly what context was sent to the model

This taught a key lesson: **retrieval quality matters more than model quality** for factual accuracy in RAG systems.

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/documind.git
cd documind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key (optional — can also enter in the UI)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

### 5. Try it with the sample document
Upload `sample_docs/ai_history.txt` and ask:
- *"Who coined the term Artificial Intelligence?"*
- *"What caused the first AI winter?"*
- *"How does RAG address hallucination?"*

---

## Demo

> 📺 **Recommended YouTube resources to understand the stack:**
>
> - [**RAG from Scratch (no LangChain)**](https://youtu.be/sVcwVQRHIc8) — freeCodeCamp, ~2 hrs, covers chunking + retrieval  
> - [**Anthropic API Crash Course**](https://youtu.be/QdDoFfkVkcw) — covers streaming, multi-turn messages  
> - [**Streamlit in 12 Minutes**](https://youtu.be/JwSS70SZdyM) — get comfortable with the UI framework  
>
> **To record your own demo video (free):**  
> Use [OBS Studio](https://obsproject.com/) (free, cross-platform) — start recording, open the app, upload the sample doc, ask 3-4 questions, stop recording. A 90-second screencast is all you need.

---

## Project Structure

```
documind/
├── app.py                 # Full application (ingestion + retrieval + UI)
├── requirements.txt       # anthropic, streamlit, pdfplumber
├── README.md
└── sample_docs/
    └── ai_history.txt     # Demo document — history of AI
```

---

## Possible Extensions

- **Semantic search** — swap TF-IDF for sentence-transformers embeddings + cosine similarity
- **Multi-document** — index multiple files and tag chunks by source
- **Citation highlighting** — show which exact sentence answered the question
- **Re-ranking** — use a cross-encoder model to re-score retrieved chunks before injection

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-20250514`) |
| Retrieval | Custom TF-IDF (pure Python stdlib) |
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Chunking | Custom overlapping sliding window |

---

*Built as a portfolio project demonstrating a non-trivial LLM API integration with a hand-rolled RAG pipeline.*
