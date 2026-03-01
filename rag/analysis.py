import json
from collections import Counter
from typing import List, Dict, Set

from config.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# categories we care about
AI_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"}


def trending_topics(n: int = 10) -> List[tuple]:
    """Return top-n categories across the metadata file."""
    counts = Counter()
    try:
        with open(settings.arxiv_metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                cats = record.get("categories", "").split()
                counts.update(cats)
    except Exception as e:
        logger.error(f"Failed to compute trending topics: {e}")
    return counts.most_common(n)


def total_filtered_records() -> int:
    """Return the number of metadata records matching AI_CATEGORIES."""
    c = 0
    try:
        with open(settings.arxiv_metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                cats = {cat.strip() for cat in record.get("categories", "").split()}
                if cats & AI_CATEGORIES:
                    c += 1
    except Exception as e:
        logger.error(f"Failed to count filtered records: {e}")
    return c


def sample_papers_by_category(category: str, limit: int = 10) -> List[Dict[str, str]]:
    """Return up to `limit` papers (title, arxiv_id) that contain the category."""
    results = []
    try:
        with open(settings.arxiv_metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= limit:
                    break
                record = json.loads(line)
                cats = record.get("categories", "").split()
                if category in cats:
                    results.append({
                        "title": record.get("title", ""),
                        "arxiv_id": record.get("id", ""),
                    })
    except Exception as e:
        logger.error(f"Failed to sample papers for {category}: {e}")
    return results
