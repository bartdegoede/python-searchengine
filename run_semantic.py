import itertools
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from load import load_documents
from search.embeddings import embed_batch, embed_text, get_embedding_model
from search.timing import timing
from search.vector_index import VectorIndex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# set httpx logging to WARNING to reduce noise from API calls
logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH_SIZE = 256
CHECKPOINT_SIZE = 10_000
INDEX_PATH = "data/vector_index"
CHECKPOINT_DIR = Path("data/checkpoints")


@timing
def build_vector_index(documents, total, model):
    logger.info(f"Building index for {total} documents...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    num_chunks = (total + CHECKPOINT_SIZE - 1) // CHECKPOINT_SIZE
    matrix = None
    build_start = time.perf_counter()

    for chunk_num, i in enumerate(range(0, total, CHECKPOINT_SIZE), 1):
        chunk_path = CHECKPOINT_DIR / f"chunk_{i}.npy"
        docs_path = CHECKPOINT_DIR / f"chunk_{i}.json"
        end = min(i + CHECKPOINT_SIZE, total)
        chunk_size = end - i

        # Consume exactly one chunk from the generator. We only hold ~10k
        # Abstract objects at a time instead of all 6.4M.
        chunk_docs = list(itertools.islice(documents, chunk_size))

        # Save doc metadata per-chunk so we never need all docs in memory.
        # On resume these are read back from disk to assemble the final JSON.
        # Write to a temp file then rename so a crash mid-write can't leave
        # a corrupt file that blocks resume.
        if not docs_path.exists():
            chunk_docs_data = {
                str(i + j): {"ID": d.ID, "title": d.title, "abstract": d.abstract, "url": d.url}
                for j, d in enumerate(chunk_docs)
            }
            with tempfile.NamedTemporaryFile("w", dir=CHECKPOINT_DIR, suffix=".json", delete=False) as f:
                json.dump(chunk_docs_data, f)
                tmp_path = Path(f.name)
            tmp_path.rename(docs_path)

        t0 = time.perf_counter()
        if chunk_path.exists():
            chunk_vectors = np.load(chunk_path)
            elapsed = time.perf_counter() - t0
            logger.info(f"  Chunk {chunk_num}/{num_chunks}: loaded checkpoint ({end}/{total} docs) in {elapsed:.1f}s")
        else:
            logger.info(f"  Chunk {chunk_num}/{num_chunks}: embedding docs {i:,}–{end:,} of {total:,}")
            texts = [d.fulltext for d in chunk_docs]
            chunk_vectors = embed_batch(model, texts, batch_size=BATCH_SIZE, show_progress=True)
            elapsed = time.perf_counter() - t0
            logger.info(f"  Chunk {chunk_num}/{num_chunks}: embedded in {elapsed:.1f}s")
            # Save raw embeddings via temp file + rename for crash safety
            with tempfile.NamedTemporaryFile(dir=CHECKPOINT_DIR, suffix=".npy", delete=False) as f:
                np.save(f, chunk_vectors)
                tmp_path = Path(f.name)
            tmp_path.rename(chunk_path)

        total_elapsed = time.perf_counter() - build_start
        logger.info(f"  Total elapsed: {total_elapsed:.1f}s")

        # We can only create the memmap once we know the embedding dimensions
        # from the first chunk (e.g. 384 for all-MiniLM-L6-v2).
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                f"{INDEX_PATH}.npy", mode="w+", dtype=np.float16,
                shape=(total, chunk_vectors.shape[1]),
            )

        # Normalize in float32 for numerical stability, then downcast to float16
        # to halve disk/memory usage. The precision loss is negligible for ranking.
        norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix[i:end] = (chunk_vectors / norms).astype(np.float16)

    if matrix is not None:
        matrix.flush()
        del matrix

    # Assemble final document metadata from per-chunk JSON files
    all_docs_data = {}
    for i in range(0, total, CHECKPOINT_SIZE):
        with open(CHECKPOINT_DIR / f"chunk_{i}.json") as f:
            all_docs_data.update(json.load(f))
    with open(f"{INDEX_PATH}.json", "w") as f:
        json.dump(all_docs_data, f)

    # Load the finished index using memory-mapped I/O — the matrix stays on disk
    # and the OS pages in data as needed during search.
    index = VectorIndex()
    index.load(INDEX_PATH)
    return index


if __name__ == "__main__":
    model = get_embedding_model()

    # try loading a saved index first
    try:
        index = VectorIndex()
        index.load(INDEX_PATH)
        logger.info(f"Loaded vector index with {len(index.documents)} documents from disk")
    except FileNotFoundError:
        logger.info("No saved index found, building from scratch...")
        total, documents = load_documents()
        index = build_vector_index(documents, total, model)

    logger.info(f"Index contains {len(index.documents)} documents")

    queries = [
        "London Beer Flood",
        "alcoholic beverage disaster in England",
        "python programming language",
        "large constricting reptiles",
    ]
    for query in queries:
        print(f'\n--- Query: "{query}" ---')
        query_vector = embed_text(model, query)
        results = index.search(query_vector, k=5)
        for doc, score in results:
            print(f"  {score:.4f} | {doc.title}")
