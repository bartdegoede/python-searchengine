# python-searchengine

Simple search engine implementation in Python for illustrative purposes to go with [this blog post](https://bart.degoe.de/building-a-full-text-search-engine-150-lines-of-code/).

## Requirements

Python 3.10 or greater, and [uv](https://docs.astral.sh/uv/).

## Usage

Install dependencies:

```bash
uv sync
```

Run the full-text search from the command line. On first run, the Wikipedia dataset (~20GB) will be downloaded from [Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia) and cached automatically:

```bash
uv run python run.py
```

Run the semantic (vector) search:

```bash
uv run python run_semantic.py
```

On first run this builds a vector index by embedding all 6.4M documents. Embeddings are checkpointed to `data/checkpoints/` so you can resume if interrupted. The finished index is saved to `data/vector_index.*` and memory-mapped on subsequent runs.

If you'd like to download the dataset separately (e.g. before a demo):

```bash
uv run python download.py
```

To get higher download rate limits, set a [Hugging Face token](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN=hf_...
```

Run from interactive console:

```python
uv run ipython

In [1]: run run.py
In [2]: index.search('python programming language', rank=True)[:5]
```

## Development

Lint and type check:

```bash
uv run ruff check .
uv run mypy search/
```

Run tests:

```bash
uv run pytest -v
```
