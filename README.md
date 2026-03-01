# MyStudyPal: arXiv RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot built on the arXiv dataset. Users can ask questions about AI research papers and receive grounded answers with citations.

## 🌟 Features

- **Knowledge base from arXiv metadata** filtered by AI-relevant categories
- Optional PDF download and ingestion
- Retrieval augmented generation with LangChain & Groq LLM
- Multi-turn conversation with session memory
- Streamlit UI with debug, configuration, and evaluation
- Retrieval explainability and metrics
- Research & comparison modes
- Trending topic analysis and simple implementation finder

## 🗂 Project Structure

```
app/
  main.py             # Streamlit entrypoint and UI logic
  ui/                 # (future) UI components
rag/
  ingest.py           # ingestion pipeline for metadata & PDFs
  retriever.py        # vector store wrapper
  chain.py            # RAG chain factory
  analysis.py         # dataset analytics utilities
config/
  config.py           # environment-based settings
data/                 # raw data (metadata, PDFs)
vectorstore/         # chroma persistence
utils/
  logger.py           # simple logging helper
  arxiv_utils.py      # pdf download helper
.env.example
requirements.txt
README.md
```

## 🛠 Tech Stack

- Python 3.13
- Streamlit for UI
- LangChain for LLM orchestration
- ChromaDB (local) for vector storage
- HuggingFace `sentence-transformers` embeddings
- Groq LLM via `langchain-groq`
- `unstructured` for PDF parsing
- `nltk` for text preprocessing (future use)

## 🚀 Getting Started

1. **Clone repo**
2. **Create (or activate) a Python virtual environment** and install dependencies:

   ```bash
   cd /path/to/MyStudyPal
   python -m venv venv          # or `venv1`/`venv2` if you prefer
   source venv/bin/activate     # macOS/Linux
   # windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Alternatively you can use the provided `venv1` which is already configured by the tools:

   ```bash
   source venv1/bin/activate
   ```

3. **Copy `.env.example` to `.env`** and populate your Groq API key.
4. **Prepare data**: place `arxiv-metadata-oai-snapshot.json` in `data/` or update path in `.env`.
5. **Run ingestion** (either via CLI or in the UI sidebar).

```bash
# build embedding store (use --force to rebuild)
python -m rag.ingest --path data/arxiv-metadata-oai-snapshot.json --persist ./vectorstore/chroma [--force]
# then start the UI (ensure the venv is active):
streamlit run app/main.py
```

6. **Start chatting!**

## ✅ Sample Queries

- "What are the latest advances in self-supervised learning?"
- "Compare BERT and GPT in natural language understanding."
- "Do any papers provide code on GitHub?"
- "Give me a literature summary on graph neural networks."


## 🔮 Future Improvements

- Persistent database for chat history (SQLite / Redis)
- More sophisticated reranking / context compression
- Automatic PDF scraping and scheduled updates
- Integration with arXiv API for real-time ingestion
- Topic modeling and better trending visualization
- Export answers to Markdown or PDF
- Authentication and user profiles for saved sessions

