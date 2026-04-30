"""Dataset loading and sampling."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_process_books(
    csv_filename: str,
    min_books_per_genre: int = 20,
    books_per_genre: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the books CSV, drop incomplete rows, then sample up to
    ``books_per_genre`` titles from each genre that has at least
    ``min_books_per_genre`` entries.

    Returns a deterministic DataFrame with columns:
    ``title``, ``genre``, ``summary``, ``text``.
    """
    df = pd.read_csv(csv_filename, index_col=None, usecols=["title", "genre", "summary"])
    logger.info("Loaded %d rows from %s", len(df), csv_filename)

    df = (
        df[df["summary"].notna()]
        .drop_duplicates(subset=["title"])
        .reset_index(drop=True)
    )
    logger.info("%d rows after cleaning", len(df))

    df["text"] = df["title"] + " " + df["summary"] + " " + df["genre"]

    popular = df["genre"].value_counts()
    popular = popular[popular >= min_books_per_genre].index
    df = df[df["genre"].isin(popular)]

    sampled = (
        df.groupby("genre", group_keys=False)
        .apply(lambda g: g.sample(n=min(books_per_genre, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )
    sampled["text"] = sampled["text"].astype(str)
    logger.info(
        "Sampled %d books across %d genres", len(sampled), sampled["genre"].nunique()
    )
    return sampled
