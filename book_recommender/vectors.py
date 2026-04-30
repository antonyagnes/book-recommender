"""ChromaDB vector store + sentence-transformer embedder."""
from __future__ import annotations

import logging
import os
from typing import Tuple

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Quiet down Chroma telemetry before any client is created.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

logger = logging.getLogger(__name__)


def build_vectors(
    df: pd.DataFrame,
    embedder_model: str = "all-MiniLM-L6-v2",
    collection_name: str = "books",
    persist_directory: str = "./data/chroma_db",
    rebuild: bool = False,
) -> Tuple[SentenceTransformer, chromadb.Collection]:
    """Return a ``(embedder, collection)`` pair, loading the existing Chroma
    collection from disk when possible. Pass ``rebuild=True`` to drop and
    recreate the collection from ``df``.
    """
    logger.info("Loading embedder: %s", embedder_model)
    embedder = SentenceTransformer(embedder_model, device="cpu")

    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )

    existing = {col.name for col in client.list_collections()}
    if collection_name in existing:
        if rebuild:
            logger.info("Rebuilding collection '%s'", collection_name)
            client.delete_collection(collection_name)
        else:
            collection = client.get_collection(collection_name)
            logger.info(
                "Loaded existing collection '%s' (%d docs)",
                collection_name,
                collection.count(),
            )
            return embedder, collection

    logger.info("Creating collection '%s' with %d documents", collection_name, len(df))
    collection = client.create_collection(collection_name)

    # Batch-encode for speed.
    texts = df["text"].tolist()
    ids = [str(i) for i in df.index.tolist()]
    metadatas = [
        {"title": str(row["title"]), "genre": str(row["genre"])}
        for _, row in df.iterrows()
    ]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    logger.info("Collection ready (%d docs)", collection.count())
    return embedder, collection
