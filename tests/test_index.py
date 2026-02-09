from search.documents import Abstract
from search.index import Index


def _make_abstract(id, title, abstract):
    return Abstract(ID=id, title=title, abstract=abstract, url=f"https://example.com/{id}")


def _build_index():
    index = Index()
    docs = [
        _make_abstract(1, "Python programming", "Python is a programming language"),
        _make_abstract(2, "Java programming", "Java is a programming language"),
        _make_abstract(3, "Python snakes", "Pythons are large snakes found worldwide"),
    ]
    for doc in docs:
        index.index_document(doc)
    return index


class TestAbstract:
    def test_fulltext(self):
        doc = _make_abstract(1, "Hello", "World")
        assert doc.fulltext == "Hello World"

    def test_term_frequency(self):
        doc = _make_abstract(1, "Python programming", "Python is a programming language")
        doc.analyze()
        assert doc.term_frequency("python") > 0
        assert doc.term_frequency("nonexistent") == 0


class TestIndex:
    def test_index_document(self):
        index = Index()
        doc = _make_abstract(1, "Test title", "Test abstract")
        index.index_document(doc)
        assert 1 in index.documents
        assert len(index.documents) == 1

    def test_index_document_no_duplicate(self):
        index = Index()
        doc = _make_abstract(1, "Test title", "Test abstract")
        index.index_document(doc)
        index.index_document(doc)
        assert len(index.documents) == 1

    def test_document_frequency(self):
        index = _build_index()
        # "program" (stemmed) appears in docs 1 and 2
        assert index.document_frequency("program") >= 2

    def test_search_and(self):
        index = _build_index()
        results = index.search("Python programming", search_type="AND")
        # Only doc 1 has both "python" and "programming" stemmed tokens
        ids = {doc.ID for doc in results}
        assert 1 in ids
        # Doc 3 has "python" but not "programming"
        assert 3 not in ids

    def test_search_or(self):
        index = _build_index()
        results = index.search("Python programming", search_type="OR")
        ids = {doc.ID for doc in results}
        # All three docs should match (python or programming)
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids

    def test_search_invalid_type(self):
        index = _build_index()
        results = index.search("Python", search_type="INVALID")
        assert results == []

    def test_search_ranked(self):
        index = _build_index()
        results = index.search("Python programming", search_type="AND", rank=True)
        # Ranked results are tuples of (document, score)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)
            assert score > 0

    def test_search_ranked_ordering(self):
        index = _build_index()
        results = index.search("Python programming", search_type="OR", rank=True)
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
