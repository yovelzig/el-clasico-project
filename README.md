# El Clasico AI — Setup & Run Guide

## Prerequisites
- Python 3.9+
- Gemini API key (get one at https://aistudio.google.com)

## Installation

```bash
cd elclasico
pip install -r requirements.txt
```

## Environment Variables

Set your Gemini API key before running:

**Linux/Mac:**
```bash
export Gemini_API_KEY="your_gemini_api_key"
```

**Windows (PowerShell):**
```powershell
$env:Gemini_API_KEY="your_gemini_api_key"
```

**Windows (Command Prompt):**
```cmd
set Gemini_API_KEY="your_gemini_api_key"
```

## Run

```bash
python app.py
```

Visit: http://localhost:5000

## First Run
On first launch, the app will:
1. Load all `.txt` files from the `data/` folder
2. Build a TF-IDF vector index (saved to `database/`)
3. Start the Flask server

Subsequent runs load the cached index instantly (much faster).

## Architecture

```
elclasico/
├── app.py              # Flask application & API routes
├── database.py         # SQLite chat history (sessions + messages)
├── rag/
│   ├── embeddings.py   # TF-IDF vectorizer (local, no downloads)
│   ├── vector_store.py # FAISS index build/load with caching
│   ├── retrieval.py    # Semantic search over the index
│   └── llm.py          # Claude claude-sonnet-4-20250514 LLM calls
├── templates/
│   └── index.html      # Full premium frontend UI
├── data/               # El Clasico football .txt data files
├── database/           # SQLite DB + FAISS index cache
└── requirements.txt
```

## RAG Pipeline
1. User question → TF-IDF embedding
2. FAISS nearest-neighbor search over 98 document chunks
3. Top-5 relevant chunks retrieved
4. Claude claude-sonnet-4-20250514 answers using context + conversation history
5. Falls back to general knowledge if documents don't contain the answer

## Chat Features
- Semantic RAG over all 7 football data files
- Conversation memory stored in SQLite (per-session)
- New chat button to reset session
- Suggested questions for quick start
- Enter key or send button to submit
