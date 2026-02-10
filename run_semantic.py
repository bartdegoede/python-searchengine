import logging
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
def build_vector_index(documents, model):
    docs = []
    texts = []
    for doc in documents:
        docs.append(doc)
        texts.append(doc.fulltext)

    logger.info(f"Loaded {len(docs)} documents, generating embeddings...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    all_vectors = []
    total = len(texts)
    num_chunks = (total + CHECKPOINT_SIZE - 1) // CHECKPOINT_SIZE
    for chunk_num, i in enumerate(range(0, total, CHECKPOINT_SIZE), 1):
        chunk_path = CHECKPOINT_DIR / f"chunk_{i}.npy"
        end = min(i + CHECKPOINT_SIZE, total)

        if chunk_path.exists():
            logger.info(f"  Chunk {chunk_num}/{num_chunks}: loading checkpoint ({end}/{total} docs)")
            chunk_vectors = np.load(chunk_path)
        else:
            logger.info(f"  Chunk {chunk_num}/{num_chunks}: embedding docs {i:,}–{end:,} of {total:,}")
            chunk_vectors = embed_batch(model, texts[i:end], batch_size=BATCH_SIZE, show_progress=True)
            np.save(chunk_path, chunk_vectors)

        all_vectors.append(chunk_vectors)

    vectors = np.vstack(all_vectors)
    index = VectorIndex(dimensions=vectors.shape[1])
    index.build(docs, vectors)
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
        index = build_vector_index(load_documents(), model)
        index.save(INDEX_PATH)
        logger.info(f"Saved vector index to {INDEX_PATH}")

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
