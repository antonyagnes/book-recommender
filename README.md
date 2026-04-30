# Lightweight Book Recommender

A book recommendation system using embeddings and vector search with ChromaDB, powered by Ollama for natural language recommendations.

## Features

- **Interactive Web UI** with Gradio for easy book discovery
- **Genre Browser** - See all available genres in your collection
- **Vector embeddings** using Sentence Transformers for semantic book search
- **Persistent storage** with ChromaDB (no need to rebuild embeddings every time)
- **LLM-powered recommendations** using Ollama models
- **Flexible data processing** with configurable genre filtering and sampling

## Requirements

- Python ^3.10
- [Ollama](https://ollama.ai/) installed and running
- At least one Ollama model installed (default: `llama3.2`)

## Installation

1. Install dependencies:
```bash
poetry install
```

2. Install Ollama:
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai/
```

3. Pull an Ollama model:
```bash
ollama pull llama3.2
```

## Usage

### Launch the Web Interface

```bash
poetry run python main.py
```

This will:
1. Load and process the book dataset
2. Build/load the vector database
3. Launch a web interface at `http://localhost:7860`

Open your browser and navigate to the URL to start getting recommendations!

## Evaluation (Retrieval + Hallucinations)

### Generate the 20-case test set

```bash
python3 eval/generate_test_set.py
```

This writes `eval/test_set.jsonl` (20 queries + relevant book titles).

### Retrieval Quality (Recall / Precision / MRR)

```bash
python eval/evaluate.py --k 10 --n-results 10
```

- **Recall@k**: % of relevant books found in top-k
- **Precision@k**: % of retrieved books in top-k that are relevant
- **MRR@k**: mean reciprocal rank of the first relevant book (higher is better)

### Recommendation Quality (Hallucination Rate with TinyLettuce)

TinyLettuce is provided by `lettucedetect` (added to `pyproject.toml`).
You can run end-to-end evaluation (retrieve → generate → TinyLettuce) with:

```bash
python eval/evaluate.py --with-generation --with-tinylettuce --ollama-model llama3.2 --k 10 --n-results 10
```

Or score a single model answer against the retrieved context via:

```python
from eval.hallucination import score_with_tinylettuce

result = score_with_tinylettuce(
    context_docs=[...],  # the retrieved docs you passed to the LLM
    question="your user query",
    answer="the LLM's recommendation text",
)
print(result.hallucination_rate_recommendations)
```

### Using the Interface

1. **Browse Genres** - The left panel shows all available genres in your collection
2. **Ask Questions** - Type your book preferences in natural language
3. **Adjust Results** - Use the slider to control how many books to search
4. **Get Recommendations** - Click the button and receive AI-powered suggestions!

**Example queries:**
- "I want a thrilling mystery with unexpected twists"
- "Looking for a romance set in historical times"
- "Science fiction about space exploration and aliens"
- "Something similar to Pride and Prejudice"

### Key Features

**Persistent Vector Database:**
- First run: builds embeddings and saves to `./data/chroma_db/` (~30 seconds)
- Subsequent runs: loads existing embeddings instantly!
- Set `rebuild=True` in code to force rebuild the collection

**Customizable Parameters:**

```python
# Load and process books
sampled_books = load_and_process_books(
    csv_filename="data/data.csv",
    min_books_per_genre=20,  # Minimum books per genre
    books_per_genre=20,      # Books to sample per genre
    random_state=42,         # For reproducibility
)

# Build or load vectors
embedder, collection = build_vectors(
    df=sampled_books,
    embedder_model="all-MiniLM-L6-v2",
    collection_name="books",
    persist_directory="./data/chroma_db",
    rebuild=False,  # Set True to delete and rebuild
)

# Get recommendations
recommendation = recommend_books(
    embedder=embedder,
    collection=collection,
    query="I want a science fiction book about space exploration",
    ollama_model="llama3.2",  # Choose your Ollama model
    n_results=10,              # Number of similar books to retrieve
)
```

## Project Structure

- `main.py` - Main application with Gradio web interface
- `data/data.csv` - Book dataset
- `data/chroma_db/` - Persistent vector database (created automatically)
- `pyproject.toml` - Poetry dependencies
- `eval/` - Test set + evaluation scripts

### Sharing Your Interface

To create a public link that others can access:

```python
interface.launch(share=True)  # Creates a temporary public URL
```

## Troubleshooting

**Ollama not found:**
- Make sure Ollama is installed and running
- Test with: `ollama run llama3.2`

**Model not available:**
- Pull the model: `ollama pull llama3.2`
- Or change to a different model in the code

**Port already in use:**
- Change the port: `interface.launch(server_port=7861)`
