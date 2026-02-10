import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .documents import Abstract
from .timing import timing


class VectorIndex:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.documents: dict[int, Abstract] = {}
        self._matrix: npt.NDArray[np.float32] | None = None

    def build(
        self, documents: Iterable[Abstract], vectors: npt.NDArray[np.float32]
    ) -> None:
        """Store documents and their pre-computed embedding vectors."""
        for i, doc in enumerate(documents):
            self.documents[i] = doc

        self._matrix = np.array(vectors, dtype=np.float32)
        # normalize all vectors to unit length so dot product = cosine similarity
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self._matrix /= norms

    @timing
    def search(
        self, query_vector: npt.NDArray[np.float32], k: int = 10
    ) -> list[tuple[Abstract, float]]:
        """Find the k documents most similar to the query vector."""
        if self._matrix is None:
            raise ValueError("Index not built. Call build() first.")
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Compute cosine similarity for all documents via dot product (vectors are normalized)
        # This is the most expensive step, but it can be done efficiently with matrix multiplication.
        scores = self._matrix @ query
        k = min(k, len(self.documents))
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(self.documents[int(i)], float(scores[i])) for i in top_k]

    def save(self, path: str | Path) -> None:
        """Save the vector index to disk.

        Creates two files:
            - {path}.npy: the normalized embedding matrix
            - {path}.json: document metadata
        """
        if self._matrix is None:
            raise ValueError("Index not built. Call build() first.")
        path = Path(path)
        np.save(f"{path}.npy", self._matrix)

        docs_data = {
            str(i): {
                "ID": doc.ID,
                "title": doc.title,
                "abstract": doc.abstract,
                "url": doc.url,
            }
            for i, doc in self.documents.items()
        }
        with open(f"{path}.json", "w") as f:
            json.dump(docs_data, f)

    def load(self, path: str | Path) -> None:
        """Load a vector index from disk using memory-mapped I/O.

        The embedding matrix is memory-mapped, so it doesn't need to fit in RAM.
        The OS will page in data from disk as needed during search.
        """
        path = Path(path)
        self._matrix = np.load(f"{path}.npy", mmap_mode="r")

        with open(f"{path}.json") as f:
            docs_data = json.load(f)

        self.documents = {
            int(i): Abstract(
                ID=d["ID"], title=d["title"], abstract=d["abstract"], url=d["url"]
            )
            for i, d in docs_data.items()
        }
