from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .documents import Abstract
from .timing import timing


class VectorIndex:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.documents: dict[int, Abstract] = {}
        self._matrix: npt.NDArray[np.float32] | None = None

    def build(self, documents: Iterable[Abstract], vectors: npt.NDArray[np.float32]) -> None:
        """Store documents and their pre-computed embedding vectors."""
        for i, doc in enumerate(documents):
            self.documents[i] = doc

        self._matrix = np.array(vectors, dtype=np.float32)
        # normalize all vectors to unit length so dot product = cosine similarity
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self._matrix /= norms

    @timing
    def search(self, query_vector: npt.NDArray[np.float32], k: int = 10) -> list[tuple[Abstract, float]]:
        """Find the k documents most similar to the query vector."""
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores = self._matrix @ query
        k = min(k, len(self.documents))
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(self.documents[int(i)], float(scores[i])) for i in top_k]
