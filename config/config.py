import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
# Also try loading from src/.env if root .env doesn't exist or doesn't have the key
if not os.getenv("GROQ_API_KEY"):
    load_dotenv("src/.env")


@dataclass
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vectorstore/chroma")
    arxiv_metadata_path: str = os.getenv("ARXIV_METADATA_PATH", "./data/arxiv-metadata-oai-snapshot.json")
    top_k: int = int(os.getenv("TOP_K", "5"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    model_name: str = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


settings = Settings()
