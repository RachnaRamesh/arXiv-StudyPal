import os
import requests
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

PDF_BASE = "https://arxiv.org/pdf/{id}.pdf"


def download_pdf(arxiv_id: str, save_dir: str = "./data/pdfs") -> Optional[str]:
    """Download a paper PDF by its arXiv id. Returns path or None on failure."""
    os.makedirs(save_dir, exist_ok=True)
    url = PDF_BASE.format(id=arxiv_id)
    out_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
    if os.path.exists(out_path):
        logger.info(f"PDF already downloaded: {out_path}")
        return out_path
    try:
        logger.info(f"Downloading PDF {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved PDF to {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"Failed to download {arxiv_id}: {e}")
        return None
