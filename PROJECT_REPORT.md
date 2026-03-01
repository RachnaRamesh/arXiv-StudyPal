# 📚 arXiv StudyPal: Project Deep-Dive & Interview Guide

This report provides a comprehensive breakdown of the **arXiv StudyPal** project, an end-to-end RAG (Retrieval-Augmented Generation) application designed for academic research.

---

## 🏗️ 1. System Architecture

The project follows a classic **RAG Architecture**, which enhances LLM responses by providing them with relevant, retrieved context from a private dataset (arXiv).

1.  **Ingestion Phase**: Raw JSON data → Cleaning → Chunking → Embedding → Vector DB (ChromaDB).
2.  **Retrieval Phase**: User Query → Vector Search (filtered by selected papers) → Top-K relevant chunks.
3.  **Generation Phase**: Query + Context + Chat History → Prompt Engineering → LLM (Llama 3.1) → Structured Answer.

---

## 📁 2. File-by-File Breakdown

### **Core Application (Frontend)**

- [main.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/app/main.py): The **heart of the UI**. Manages the Streamlit layout, multi-chat session state, sidebar navigation, and the chat loop. It also implements the "Dynamic Quick Follow-ups" logic.
- [components.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/app/ui/components.py): Contains reusable UI components. It handles the **persistence logic** (loading/saving chats to `chats.json`) and initializes the session state.

### **RAG Logic (Backend)**

- [ingest.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/rag/ingest.py): The **data architect**. It reads the 4.9GB arXiv dataset, filters for AI categories, chunks the text using `RecursiveCharacterTextSplitter`, and populates the ChromaDB vector store.
- [retriever.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/rag/retriever.py): The **search engine**. Custom class that queries ChromaDB. It’s unique because it applies a **metadata filter** to ensure we only search within the papers you've selected.
- [chain.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/rag/chain.py): The **brain**. It assembles the LangChain `ConversationalRetrievalChain`. It defines the "Expert Researcher" prompt and connects the Retriever to the Groq LLM.
- [analysis.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/rag/analysis.py): Provides utility functions to sample and categorize papers from the raw dataset for the UI dropdowns.

### **Configuration & Utils**

- [config.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/config/config.py): Centralized settings. Loads `.env` variables and sets global defaults like temperature, model names, and chunk sizes.
- [arxiv_utils.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/utils/arxiv_utils.py): Helpers for cleaning arXiv text and handling the specific formatting of the Kaggle dataset.
- [logger.py](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/utils/logger.py): Standardized logging to track errors and ingestion progress.

### **Deployment & Meta**

- [requirements.txt](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/requirements.txt): Lists all Python dependencies.
- [.streamlit/config.toml](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/.streamlit/config.toml): Customizes the Streamlit theme (colors, fonts) and server settings.
- [.env.example](file:///Users/rachnaramesh/Desktop/Projects/AI/MyStudyPal/.env.example): A template for other developers to know which API keys they need.

---

## 🧠 3. Key Technical Decisions (Interview Gold)

- **Why Groq?**: We used Groq (Llama-3.1-8b) for **speed**. It provides near-instant inference, which is critical for a good chat UX.
- **Why ChromaDB?**: It’s an **open-source, serverless** vector database that allows us to store embeddings locally without needing a complex database setup.
- **Why Recursive Chunking?**: Unlike fixed-size chunking, `RecursiveCharacterTextSplitter` tries to keep paragraphs and sentences together, preserving the semantic meaning of scientific text.
- **Persistence Strategy**: We used a local `chats.json` for persistence. While a production app might use MongoDB, JSON is a lightweight and effective solution for a MVP/Portfolio project.

---

## ❓ 4. Common Interview Questions

**Q: "How do you handle the LLM hallucinating about paper content?"**

> _A: By using RAG. We strictly instruct the model in the system prompt to only use the provided context. If the answer isn't in the context, it must say 'I don't have enough information'._

**Q: "What happens if I select 50 papers at once?"**

> _A: The retriever still only pulls the top-K (e.g., 5) most relevant chunks across all 50 papers. This keeps the prompt within the LLM's context window limit while still covering all relevant documents._

**Q: "How did you optimize the UI for research?"**

> _A: I implemented dynamic follow-up questions that change based on the conversation stage (Discovery → Deep Dive → Synthesis), guiding the user through the scientific method._

---

## 🚀 5. Deployment Summary

The project is deployed on **Streamlit Cloud**, connected to a **GitHub** repository. Secrets (API keys) are managed via Streamlit's encrypted secrets manager, ensuring no sensitive data is exposed in the code.
