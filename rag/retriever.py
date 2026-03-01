from typing import Any, List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config.config import settings


class ArxivRetriever:
    def __init__(self, persist_directory: str = None, embedding_model: str = None):
        self.persist_directory = persist_directory or settings.vector_store_path
        self.embedding_model = embedding_model or settings.embedding_model
        self._vectordb = None

    def _load_vectorstore(self) -> Chroma:
        if self._vectordb is None:
            embed = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self._vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embed,
            )
        return self._vectordb

    def search(self, query: str, top_k: int = None, arxiv_ids: List[str] = None) -> List[Tuple[Any, float]]:
        """Return list of (doc, score)"""
        top_k = top_k or settings.top_k
        vectordb = self._load_vectorstore()
        
        search_kwargs = {"k": top_k}
        if arxiv_ids:
            search_kwargs["filter"] = {"arxiv_id": {"$in": arxiv_ids}}
            
        docs = vectordb.similarity_search_with_score(query, **search_kwargs)
        return docs

    def count_docs(self) -> int:
        """Return number of indexed chunks in the vectorstore."""
        vectordb = self._load_vectorstore()
        try:
            return vectordb._collection.count()
        except Exception:
            return 0
