# python-searchengine

Simple search engine implementation in Python for illustrative purposes to go with [this blog post](https://bart.degoe.de/building-a-full-text-search-engine-150-lines-of-code/).

## Requirements

Python 3.10 or greater, and [uv](https://docs.astral.sh/uv/).

## Usage

Install dependencies:

```bash
uv sync
```

Run from the command line. On first run, the Wikipedia dataset (~20GB) will be downloaded from [Hugging Face](https://huggingface.co/datasets/wikimedia/wikipedia) and cached automatically:

```bash
uv run python run.py
# loads of log output
index_documents took 1714.3159050941467 seconds
Index contains 6407814 documents
search took 0.3170650005340576 seconds
search took 4.130218982696533 seconds
search took 0.005632877349853516 seconds
search took 17.051696300506592 seconds
```

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

Lint with ruff:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest -v
```
