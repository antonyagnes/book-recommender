"""Launch the Gradio book-recommender UI.

The pipeline is implemented in the ``book_recommender`` package; this module
is just the thin entrypoint that wires it up.
"""
from __future__ import annotations

import logging
import os

# Disable Chroma telemetry as early as possible.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

from book_recommender.app import create_gradio_interface, get_available_genres
from book_recommender.data import load_and_process_books
from book_recommender.vectors import build_vectors

logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def main() -> None:
    books_per_genre = 50
    df = load_and_process_books(
        csv_filename="data/data.csv",
        min_books_per_genre=20,
        books_per_genre=books_per_genre,
    )
    embedder, collection = build_vectors(
        df=df,
        embedder_model="all-MiniLM-L6-v2",
        collection_name="books",
        persist_directory="./data/chroma_db",
        rebuild=False,
    )
    interface = create_gradio_interface(
        embedder=embedder,
        collection=collection,
        available_genres=get_available_genres(df),
        books_per_genre=books_per_genre,
        ollama_model="llama3.2:3b",
    )
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
