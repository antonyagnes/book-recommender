"""Gradio UI for the book recommender."""
from __future__ import annotations

from typing import List

import chromadb
import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer

from book_recommender.recommender import recommend_books


def get_available_genres(df: pd.DataFrame) -> List[str]:
    return sorted(df["genre"].unique().tolist())


def create_gradio_interface(
    embedder: SentenceTransformer,
    collection: chromadb.Collection,
    available_genres: List[str],
    books_per_genre: int = 10,
    ollama_model: str = "llama3.2:3b",
) -> gr.Blocks:
    def _recommend(user_query: str, num_results: int) -> str:
        if not user_query.strip():
            return "Please enter a query to get recommendations."
        try:
            return recommend_books(
                embedder=embedder,
                collection=collection,
                query=user_query,
                num_books=num_results,
                ollama_model=ollama_model,
            )
        except Exception as exc:
            return f"Error: {exc}\n\nMake sure Ollama is running with model '{ollama_model}'."

    with gr.Blocks(title="Lightweight Book Recommender", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI-Powered Lightweight Book Recommender")
        gr.Markdown(
            "Get personalised book recommendations using semantic search + a local LLM.\n\n"
            "Powered by ChromaDB and Ollama."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Available Genres")
                gr.Markdown(f"*Up to {books_per_genre} books per genre*")
                gr.Textbox(
                    value="\n\n".join(f"• {g}" for g in available_genres),
                    lines=20,
                    max_lines=20,
                    label=f"Total: {len(available_genres)} genres",
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("### Find Your Next Great Read")
                query_input = gr.Textbox(
                    label="What kind of book are you looking for?",
                    placeholder="e.g. 'A mystery novel with a strong female detective'",
                    lines=3,
                )
                num_results = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Number of recommendations",
                )
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
                output = gr.Markdown(value="Your recommendations will appear here…")

                recommend_btn.click(
                    fn=_recommend,
                    inputs=[query_input, num_results],
                    outputs=output,
                )

                gr.Examples(
                    examples=[
                        ["A thrilling mystery with unexpected twists"],
                        ["Romance set in historical times"],
                        ["Science fiction about space exploration"],
                        ["Fantasy with magic and epic adventures"],
                        ["Something similar to Pride and Prejudice"],
                    ],
                    inputs=query_input,
                )

        gr.Markdown(
            "---\n"
            "**How it works:** your query is embedded → similar books are retrieved "
            "via vector search → a local LLM selects and explains the best matches."
        )

    return interface
