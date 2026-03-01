import os
from typing import Optional, List

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate

from config.config import settings
from rag.retriever import ArxivRetriever


def get_groq_llm(model_name: str = "llama-3.1-8b-instant", temperature: float = None):
    temperature = temperature if temperature is not None else settings.temperature
    # Ensure api_key is passed correctly to ChatGroq
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in settings or environment")
    return ChatGroq(groq_api_key=api_key, model=model_name, temperature=temperature)


def make_prompt(mode: str = "normal") -> PromptTemplate:
    base = (
        "You are an expert academic researcher. Provide a highly organized, professional summary of the selected papers.\n\n"
        "STRICT FORMATTING RULES:\n"
        "1. Start with a brief, high-level summary (1-2 sentences).\n"
        "2. Use **Markdown headings** (###) for distinct sections like 'Key Findings', 'Methodology', 'Results', etc. Do not use # or ## headings.\n"
        "3. Use **bullet points** extensively for clarity.\n"
        "4. **Bold** key technical terms or specific metrics.\n"
        "5. Always cite the paper title and year in brackets [Paper Title, Year].\n"
        "6. Ensure the output is visually clean with proper spacing between sections.\n\n"
    )
    if mode == "research":
        base += (
            "### Research Synthesis\n"
            "Provide a comprehensive literature summary, grouping similar findings and highlighting open questions.\n\n"
        )
    elif mode == "compare":
        base += (
            "### Comparative Analysis\n"
            "Directly compare the papers' approaches, strengths, and weaknesses. Use a comparison table or a structured list.\n\n"
        )
    template = base + (
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (Markdown format):"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def create_conversational_chain(
    retriever: Optional[ArxivRetriever] = None,
    llm_model: str = "llama-3.1-8b-instant",
    temperature: float = None,
    mode: str = "normal",
    arxiv_ids: List[str] = None,
) -> ConversationalRetrievalChain:
    retriever = retriever or ArxivRetriever()
    llm = get_groq_llm(model_name=llm_model, temperature=temperature)

    prompt = make_prompt(mode=mode)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    search_kwargs = {"k": settings.top_k}
    if arxiv_ids:
        search_kwargs["filter"] = {"arxiv_id": {"$in": arxiv_ids}}

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever._load_vectorstore().as_retriever(search_kwargs=search_kwargs),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain
