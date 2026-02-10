import numpy as np

from search.documents import Abstract
from search.vector_index import VectorIndex


def _make_abstract(id, title, abstract):
    return Abstract(ID=id, title=title, abstract=abstract, url=f"https://example.com/{id}")


def _build_vector_index():
    """Build a small vector index with 4 documents and fake 4-dim embeddings."""
    docs = [
        _make_abstract(0, "London Beer Flood", "A flood of beer in London in 1814"),
        _make_abstract(1, "Boston Molasses Flood", "A flood of molasses in Boston in 1919"),
        _make_abstract(2, "Python programming", "Python is a programming language"),
        _make_abstract(3, "Java programming", "Java is a programming language"),
    ]
    # fake embeddings: docs 0 and 1 are similar, docs 2 and 3 are similar
    vectors = np.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ], dtype=np.float32)

    index = VectorIndex(dimensions=4)
    index.build(docs, vectors)
    return index


class TestVectorIndex:
    def test_build(self):
        index = _build_vector_index()
        assert len(index.documents) == 4

    def test_search_returns_k_results(self):
        index = _build_vector_index()
        query = np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)
        assert len(results) == 2

    def test_search_returns_tuples(self):
        index = _build_vector_index()
        query = np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)
        for doc, score in results:
            assert isinstance(doc, Abstract)
            assert isinstance(score, float)

    def test_search_similarity_ranking(self):
        index = _build_vector_index()
        # query similar to "flood" docs (dims 0-1)
        query = np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=4)
        # flood docs should rank higher than programming docs
        ids = [doc.ID for doc, _ in results]
        assert ids[0] in (0, 1)
        assert ids[1] in (0, 1)

    def test_search_scores_descending(self):
        index = _build_vector_index()
        query = np.array([1.0, 0.8, 0.1, 0.0], dtype=np.float32)
        results = index.search(query, k=4)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_index(self):
        index = _build_vector_index()
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=100)
        assert len(results) == 4


class TestVectorIndexPersistence:
    def test_save_and_load(self, tmp_path):
        index = _build_vector_index()
        index.save(tmp_path / "test_index")

        loaded = VectorIndex(dimensions=4)
        loaded.load(tmp_path / "test_index")

        assert len(loaded.documents) == 4
        assert loaded._matrix.shape == (4, 4)

    def test_loaded_index_search(self, tmp_path):
        index = _build_vector_index()
        index.save(tmp_path / "test_index")

        loaded = VectorIndex(dimensions=4)
        loaded.load(tmp_path / "test_index")

        query = np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)
        results = loaded.search(query, k=2)
        assert len(results) == 2
        ids = [doc.ID for doc, _ in results]
        assert ids[0] in (0, 1)

    def test_loaded_index_is_memmap(self, tmp_path):
        index = _build_vector_index()
        index.save(tmp_path / "test_index")

        loaded = VectorIndex(dimensions=4)
        loaded.load(tmp_path / "test_index")

        assert isinstance(loaded._matrix, np.memmap)
