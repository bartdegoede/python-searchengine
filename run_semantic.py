import logging

from load import load_documents
from search.embeddings import embed_batch, embed_text, get_embedding_model
from search.timing import timing
from search.vector_index import VectorIndex

logging.basicConfig(level=logging.INFO)
# set httpx logging to WARNING to reduce noise from API calls
logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH_SIZE = 256
INDEX_PATH = "data/vector_index"


@timing
def build_vector_index(documents, model):
    docs = []
    texts = []
    for doc in documents:
        docs.append(doc)
        texts.append(doc.fulltext)

    print(f"Loaded {len(docs)} documents, generating embeddings...")
    vectors = embed_batch(model, texts, batch_size=BATCH_SIZE, show_progress=True)

    index = VectorIndex(dimensions=vectors.shape[1])
    index.build(docs, vectors)
    return index


if __name__ == "__main__":
    model = get_embedding_model()

    # try loading a saved index first
    try:
        index = VectorIndex()
        index.load(INDEX_PATH)
        print(f"Loaded vector index with {len(index.documents)} documents from disk")
    except FileNotFoundError:
        print("No saved index found, building from scratch...")
        index = build_vector_index(load_documents(), model)
        index.save(INDEX_PATH)
        print(f"Saved vector index to {INDEX_PATH}")

    print(f"Index contains {len(index.documents)} documents")

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
