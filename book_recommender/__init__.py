"""Book recommender RAG pipeline."""
from book_recommender.data import load_and_process_books
from book_recommender.prompts import DEFAULT_PROMPT
from book_recommender.recommender import recommend_books, retrieve
from book_recommender.vectors import build_vectors

__all__ = [
    "DEFAULT_PROMPT",
    "build_vectors",
    "load_and_process_books",
    "recommend_books",
    "retrieve",
]
