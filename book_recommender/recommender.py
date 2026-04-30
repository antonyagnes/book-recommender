"""End-to-end retrieve + generate recommendation pipeline."""
from __future__ import annotations

import re
import ollama 
from dataclasses import dataclass
from typing import List, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from book_recommender.prompts import DEFAULT_PROMPT


@dataclass(frozen=True)
class Retrieved:
    titles: List[str]
    docs: List[str]
    genres: List[str]
    distances: List[float]


def retrieve(
    embedder: SentenceTransformer,
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
) -> Retrieved:
    """Vector-search the collection and return ordered titles, docs, genres,
    and distances."""
    q_emb = embedder.encode(query).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = (res.get("metadatas") or [[]])[0] or []
    docs = (res.get("documents") or [[]])[0] or []
    distances = (res.get("distances") or [[]])[0] or []

    titles, genres = [], []
    for md in metadatas:
        if isinstance(md, dict):
            titles.append(str(md.get("title", "")))
            genres.append(str(md.get("genre", "")))
        else:
            titles.append("")
            genres.append("")
    return Retrieved(titles=titles, docs=docs, genres=genres, distances=list(distances))


def _truncate_to_n_items(text: str, n: int) -> str:
    """Drop list items beyond the first ``n``."""
    if not text or n <= 0:
        return text
    lines = text.splitlines()
    item_indices = [
        i for i, line in enumerate(lines)
        if re.match(r"^(\d+[\).]|[-*•])\s+\S+", line.strip())
    ]
    if len(item_indices) <= n:
        return text
    return "\n".join(lines[: item_indices[n]]).rstrip() or text


def recommend_books(
    embedder: SentenceTransformer,
    collection: chromadb.Collection,
    query: str,
    num_books: int = 3,
    ollama_model: str = "llama3.2:3b",
    n_results: int = 10,
    prompt_template: str = DEFAULT_PROMPT,
    ollama_options: dict | None = None,
) -> str:
    """Retrieve top ``n_results`` books and ask an Ollama model to recommend
    exactly ``num_books`` of them.""" 
    retrieved = retrieve(embedder, collection, query, n_results=n_results)
    docs_blob = _format_docs(retrieved.docs)
    prompt = prompt_template.format(docs=docs_blob, query=query, num_books=num_books)

    response = ollama.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options=ollama_options or {"temperature": 0.2},
    )
    return _truncate_to_n_items(response["message"]["content"], num_books)


def _format_docs(docs: Sequence[str]) -> str:
    return "\n".join(f"- {d}" for d in docs)
